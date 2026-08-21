import json
import os
def deduplicate_jsonl_by_field(input_file_path, output_file_path=None,field_name='id'):
    seen_ids = set()
    unique_entries = []

    # Read the input file and collect unique entries based on 'id'
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                entry_id = entry.get(field_name)
                if entry_id is not None and entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    unique_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    # Write unique entries to the output file
    if not output_file_path:
        output_file_path = input_file_path
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in unique_entries:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Deduplication complete. {len(unique_entries)} unique entries written to {output_file_path}")


def get_difference_optimized(file1, file2, key_field, output_file):
    """
    根据key_field字段计算两个JSON Lines文件的差集，并返回结果。
    
    参数：
        file1: 第一个JSON Lines文件路径（基准文件）
        file2: 第二个JSON Lines文件路径（要从中减去的文件）
        key_field: 用于比较的字段名
        output_file: 差集结果的文件路径
        
    返回：
        差集结果的文件路径
    """
    if not os.path.exists(file2):
        return output_file
    # 收集两个文件中所有的key值
    keys_in_file1 = set()
    records1 = []
    with open(file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            record = json.loads(line)
            records1.append(record)
            keys_in_file1.add(record[key_field])
    
    keys_in_file2 = set()
    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            record = json.loads(line)
            keys_in_file2.add(record[key_field])

    # 计算file1中独有的key值
    unique_keys = keys_in_file1 - keys_in_file2

    # 构建差集记录列表
    difference_records = [record for record in records1 if record[key_field] in unique_keys]

    # 将差集写入新的JSON Lines文件
    def write_json_lines(file_path, records):
        """将记录列表写入JSON Lines文件"""
        with open(file_path, 'w', encoding='utf-8') as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
    write_json_lines(output_file, difference_records)

    return output_file