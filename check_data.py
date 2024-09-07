import json

def load_json_lines(file_path):
    """Load JSON data from a file with multiple JSON objects (one per line)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def find_common_entries(file1, file2):
    """Find common entries in the 'canonical_solution' field between two JSON files."""
    data1 = load_json_lines(file1)
    data2 = load_json_lines(file2)

    solutions1 = {entry['canonical_solution']: entry for entry in data1 if 'canonical_solution' in entry}
    solutions2 = {entry['canonical_solution']: entry for entry in data2 if 'canonical_solution' in entry}

    common_entries = [solutions1[sol] for sol in solutions1 if sol in solutions2]

    return common_entries

def main():
    file1 = '/home/zhangxuanming/CTG/HumanEval.jsonl'
    file2 = '/home/zhangxuanming/DataLeakage_v2/output2.jsonl'

    common_entries = find_common_entries(file1, file2)

    if common_entries:
        print("Found common canonical_solution entries:")
        for entry in common_entries:
            print(json.dumps(entry, ensure_ascii=False, indent=4))
    else:
        print("No common canonical_solution entries found.")

if __name__ == "__main__":
    main()
