import json


def read_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"文件 {file_path} 解析错误")
        return None

def save_config(file_path, config_data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(config_data, file, indent=4, ensure_ascii=False)
        print(f"配置已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存配置时发生错误: {e}")


def get_config():
    config_file = 'libs/wordladder/config.json'
    config = read_config(config_file)
    return config
    
def is_param_random():
    config = get_config()
    return config["random_param"]

def is_word_random():
    config = get_config()
    return config["random_param_rule"]["random_word_length"]

def get_random_mode_word_length():
    config = get_config()
    return config["random_param_rule"]["word_length"]

def get_specify_mode_start_word():
    config = get_config()
    return config["specify_param"]["start_word"]

def get_specify_mode_end_word():
    config = get_config()
    return config["specify_param"]["end_word"]

def get_solutions_count():
    config = get_config()
    if is_param_random():
        return config["random_param_rule"]["solutions"]
    else:
        return config["specify_param"]["solutions"]
    
def get_output_filepath():
    config = get_config()
    return config["output_file_path"]

def get_ladders_count():
    config = get_config()
    return config["max_ladder"]

def is_nl_describe():
    config = get_config()
    return config["NL_describe_output"]