'''
adapted from karpathy
https://github.com/karpathy/nanoGPT/blob/master/model.py
'''
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

VERBOSE = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, hidden, nhead, bias=False):
        super(Attention, self).__init__()
        assert hidden % nhead == 0, "hidden size should be divisible by nhead"
        self.hidden = hidden
        self.nhead = nhead
        self.dhead = hidden // nhead
        self.q_proj = nn.Linear(hidden, hidden, bias=bias)
        self.k_proj = nn.Linear(hidden, hidden, bias=bias)
        self.v_proj = nn.Linear(hidden, hidden, bias=bias)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, -1, self.dhead) # B, T, nhead, dhead
        k = self.k_proj(x).view(B, T, -1, self.dhead) # B, T, nhead, dhead
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).contiguous() # B, nhead, T, dhead
        k = k.transpose(1, 2).contiguous() # B, nhead, T, dhead

        v = self.v_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(x)

class MLP(nn.Module):
    def __init__(self, hidden, bias=False):
        super(MLP, self).__init__()
        self.ffn1 = nn.Linear(hidden, 4*hidden, bias)
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(4*hidden, hidden, bias)

    def forward(self, x):
        return self.ffn2(self.act(self.ffn1(x)))

class LayerNorm(nn.Module):
    def __init__(self, hidden, bias=False):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden)) if bias else None

    def forward(self, x):
        return F.layer_norm(x.float(), self.weight.shape, self.weight, self.bias, 1e-5).type_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden, nhead, bias=False):
        super(ResidualBlock, self).__init__()
        self.ln1 = LayerNorm(hidden, bias)
        self.attn = Attention(hidden, nhead, bias)
        self.ln2 = LayerNorm(hidden, bias)
        self.mlp = MLP(hidden, bias)
    
    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        return x + self.mlp(self.ln2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, hidden, nhead, nlayer, rope_theta=10000, bias=False):
        super(Transformer, self).__init__()
        assert bias == False, "currently bias is not supported"
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.block_size = block_size
        self.rope_theta = rope_theta
        self.model = nn.ModuleDict(
            dict(
                wte = nn.Embedding(vocab_size, hidden), # long tensor -> 3d tensor -> channel dim 쪼개
                h = nn.ModuleList([ResidualBlock(hidden, nhead, bias) for _ in range(nlayer)]),
                ln = LayerNorm(hidden, bias=bias),
            )
        )
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=bias)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(self.hidden // self.nhead, self.block_size * 2, self.rope_theta)

    def compute_loss(self, z, y, ignore_index=-100, reduction='mean'):
        z = z[..., :-1, :].contiguous().view(B*(T-1), -1) # B*T, C
        y = y.view(-1) # B*T, 1
        return F.cross_entropy(z, y, ignore_index=ignore_index, reduction=reduction)

    def forward(self, x):
        if VERBOSE: print('before wte', x.size())
        x = self.model.wte(x)
        if VERBOSE: print('after wte', x.size())
        for block in self.model.h:
            x = block(x, self.freqs_cis)
            if VERBOSE: print('after residual block', x.size())
        x = self.model.ln(x)
        if VERBOSE: print('after ln', x.size())
        x = self.lm_head(x).float() # projection to logit space and upcast, (B, T, C)
        if VERBOSE: print('after lm_head', x.size())
        return x