import re
import json

def remove_function_signature_v3(code: str) -> str:
    func_pattern = r"^\s*def\s+\w+\s*\(.*?\)\s*(->\s*.*\s*)?:\s*"
    import_pattern = r"^\s*(import\s+\w+|from\s+\w+\s+import\s+\w+)"
    
    lines = code.split('\n')
    processed_lines = []
    
    in_function_body = False
    library_declared = False

    for line in lines:
        if re.match(import_pattern, line):
            processed_lines.append(line)
            library_declared = True
            continue
        
        if re.match(func_pattern, line):
            if in_function_body:
                processed_lines.append(line)
            elif library_declared:
                in_function_body = True
                library_declared = False
                continue
            else:
                in_function_body = True
                continue
        elif not in_function_body:
            if line.strip() == "" or line.strip().startswith("#"):
                processed_lines.append(line)
            else:
                in_function_body = True
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def adjust_indentation_v3(prompt: str, completion: str) -> str:
    prompt_lines = prompt.split('\n')
    last_prompt_line = next((line for line in reversed(prompt_lines) if line.strip() != ""), "")
    indentation_match = re.match(r'(\s*)', last_prompt_line)
    indentation = indentation_match.group(1) if indentation_match else ''
    
    completion_lines = completion.split('\n')
    adjusted_lines = []
    
    for i, line in enumerate(completion_lines):
        if i == 0:
            adjusted_lines.append(indentation + line.strip())
        else:
            adjusted_lines.append(line)
    
    return '\n'.join(adjusted_lines)

def process_jsonl_v3(input_file_path, output_file_path):
    data = []

    with open(input_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    updated_data = []
    for entry in data:
        task_id = entry['task_id']
        prompt = entry['prompt']
        entry_point = entry['entry_point']
        completion = entry['completion']
        test = entry.get('test', '')

        cleaned_completion = remove_function_signature_v3(completion)
        adjusted_completion = adjust_indentation_v3(prompt, cleaned_completion)

        new_entry = {
            'task_id': task_id,
            'prompt': prompt,
            'entry_point': entry_point,
            'canonical_solution': adjusted_completion,
            'test': test,
            'final': prompt + adjusted_completion
        }
        updated_data.append(new_entry)
   
    with open(output_file_path, 'w') as file:
        for entry in updated_data:
            file.write(json.dumps(entry) + '\n')

input_file_path = '/Users/xuemuqiangu/Desktop/DataLeakage_v2/datasets/multi_leakage_source_humaneval.json'
output_file_path = '/Users/xuemuqiangu/Desktop/DataLeakage_v2/datasets/multi_leakage_source_humaneval.json'

process_jsonl_v3(input_file_path, output_file_path)

output_file_path
