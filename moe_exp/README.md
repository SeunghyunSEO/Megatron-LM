
## References

- megatron naive moe
    - [Megatron-LM/megatron/core/transformer/moe](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)
    - [Megatron-LM/docs/llama_mistral.md](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llama_mistral.md)
    - [Megatron-LM/examples/export/ptq_and_trtllm_export](https://github.com/NVIDIA/Megatron-LM/tree/772faca1f8d5030621b738cbd8e8bb2d8d28f6e6/examples/export/ptq_and_trtllm_export)
    - [mixtral inference example](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mixtral)

- dmoe
    - megablocks and llm foundry
        - [databricks/megablocks](https://github.com/databricks/megablocks)
        - [dmoe.py](https://github.com/databricks/megablocks/blob/main/megablocks/layers/dmoe.py#L18)
        - [layers/ffn.py](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/llmfoundry/models/layers/ffn.py#L470-L509)
        - [test_dmoe.py](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/tests/models/layers/test_dmoe.py#L71)
        - [moe init](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/llmfoundry/models/utils/param_init_fns.py#L341-L404)
        - [olmoe](https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/model.py#L680-L690)
    - megatron integration
        - [megatron PR 1](https://github.com/NVIDIA/Megatron-LM/pull/287)
        - [megatron PR 2](https://github.com/NVIDIA/Megatron-LM/pull/288/files)

```bash
cd /path/to/dir/Megatron-LM/moe_exp
pip install -r requirements.txt # get megablocks and grouped gemm
# NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# https://github.com/NVIDIA/Megatron-LM/issues/696#issuecomment-1987058741
# https://github.com/NVIDIA/TransformerEngine/issues/1014
```

```bash
cd /path/to/dir/Megatron-LM/
bash ./moe_exp/scripts/run_moe_from_scratch.sh
```

```bash
python -c "import transformer_engine as te; \
linear=te.pytorch.Linear; \
print(linear);"
```