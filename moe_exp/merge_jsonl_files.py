import os
import json
import argparse

import os
import json

def get_file_list(input_dir, exclude_file=None):
    """Walk through the directory and collect all .jsonl files, except the excluded one."""
    jsonl_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                
                # Skip the excluded file (validation file)
                if exclude_file and exclude_file in file:
                    continue
                
                jsonl_files.append(file_path)
    
    return jsonl_files

def process_jsonl_files_(input_dir, output_dir, output_file_name, exclude_file=None):
    """Process the list of .jsonl files and write them to a single output .jsonl file."""
    output_path = os.path.join(output_dir, f'{output_file_name}.jsonl')
    file_list = get_file_list(input_dir, exclude_file)
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # Loop over the collected files and process them
        for file_path in file_list:
            print(f"Processing {file_path}...")
            
            # Read each line from the jsonl file and write it directly to the output file
            try:
                with open(file_path, 'r', encoding='utf-8') as jsonl_file:
                    for line in jsonl_file:
                        try:
                            json_data = json.loads(line.strip())  # Load and verify the JSON line
                            out_file.write(json.dumps(json_data) + '\n')  # Write each JSON object as a line
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in file {file_path}: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    print(f"Combined data written to {output_file_name}.jsonl")

def process_jsonl_files(input_dir, output_dir, output_file_name, exclude_file=None):
    combined_data = []
    
    # Walk through the directory and find all .jsonl files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                
                # Skip the excluded file (validation file)
                if exclude_file and exclude_file in file:
                    continue
                
                print(f"Processing {file_path}...")
                
                # Read each line from the jsonl file and append it to the combined_data list
                try:
                    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
                        for line in jsonl_file:
                            line = line.strip()  # Remove leading/trailing whitespace
                            if not line:  # Skip empty lines
                                print(f"Warning: Empty line at {i} in {input_file}")
                                continue
                            try:
                                combined_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Skipping invalid JSON line in {file_path}: {e}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    # Write the combined data to a single json file
    output_path = os.path.join(output_dir, f'{output_file_name}.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(combined_data, json_file, indent=4)
        print(f"Combined data written to {output_path}")
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")

def process_valid_file(input_dir, valid_file_name, output_dir, output_file_name):
    validation_data = []
    
    valid_file_path = os.path.join(input_dir, f'{valid_file_name}.jsonl')
    print(f"Processing validation file {valid_file_path} ...")
    
    try:
        with open(valid_file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                line = line.strip()  # Remove leading/trailing whitespace
                if not line:  # Skip empty lines
                    print(f"Warning: Empty line at {i} in {input_file}")
                    continue
                try:
                    validation_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {valid_file_path}: {e}")
    except Exception as e:
        print(f"Error reading validation file {valid_file_path}: {e}")
    
    # Write the validation data to a separate json file
    valid_output_path = os.path.join(output_dir, f'{output_file_name}.json')
    try:
        with open(valid_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(validation_data, json_file, indent=4)
        print(f"Validation data written to {valid_output_path}")
    except Exception as e:
        print(f"Error writing validation data to {valid_output_path}: {e}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process jsonl files into a single json file.')
    
    parser.add_argument('--input_dir', type=str, default='data/fineweb_edu_10bt_shuffled', help="Path to the directory containing the .jsonl files.")
    parser.add_argument('--valid_file_name', type=str, default='fineweb_edu_10bt.val', help="Name of the validation .jsonl file without extension.")
    parser.add_argument('--output_file_name', type=str, default='fineweb_edu_10bt_shuffled_merged', help="Name of the output file for merged data.")
    args = parser.parse_args()

    process_jsonl_files_(args.input_dir, args.input_dir, args.output_file_name, exclude_file=args.valid_file_name)

    print("All files processed and validation data handled separately.")
