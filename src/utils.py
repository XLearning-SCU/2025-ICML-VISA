import yaml
import os
from datetime import datetime
import time
import json
import re
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# %%
def load_yaml():
    choose_EVAL_DATASET_yaml_path = os.path.join(VISA_path, "config", "EVAL_DATASET.yaml")
    with open(choose_EVAL_DATASET_yaml_path, "r") as f:
        choose_EVAL_DATASET_config = yaml.safe_load(f)
    EVAL_DATASETS_yaml_path = os.path.join(VISA_path, "config",
                                           f"{choose_EVAL_DATASET_config['EVAL_DATASET_name']}.yaml")
    with open(EVAL_DATASETS_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def output_time(start_time):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"{current_time} Program runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")

def read_json(path):
    assert os.path.exists(path)
    with open(path, 'r', encoding = 'utf-8') as file:
        return json.load(file)

def write_json(path, content):
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(content, f, ensure_ascii = False, indent = 4)

def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def R_t2i(scores_t2i, txt2img):
    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    R1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    R5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    R10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return R1, R5, R10

def out_to_typora_t2i(sim_t2i, txt2img):
    R1, R5, R10 = R_t2i(sim_t2i, txt2img)
    res = []
    res.append(f'{R1:.1f}')
    res.append(f'{R5:.1f}')
    res.append(f'{R10:.1f}')
    for i in range(3):
        if i != 2:
            print(f'{res[i]}|', end = "")
        else:
            print(f'{res[i]}')

def get_retrieval_element():
    config = load_yaml()
    ann1, ann2, image_ids, txt2img = [], [], [], {}
    short_dic = defaultdict(list)
    short_data, long_data = read_json(query_path), read_json(recap_path)
    print(f'-' * 50)
    for item in short_data:
        short_dic[item[f"{config['type']}_id"]].append(item["captions"][0])
        ann1.append(item["captions"][0])

    for item in long_data:
        ann2.append(item["captions"][0])
        image_ids.append(item[f"{config['type']}_id"])
    print(f'len(ann1) = {len(ann1)}')
    print(f'len(ann2) = {len(ann2)}')
    print(f'len(image_ids) = {len(image_ids)}')
    txt_id = 0
    for img_id, (key, value) in enumerate(short_dic.items()):
        for i, caption in enumerate(value):
            txt2img[txt_id] = img_id
            txt_id += 1
    print(f'-' * 50)
    assert len(ann1) > 0 and len(ann2) > 0
    assert len(image_ids) > 0 and len(txt2img) > 0
    return ann1, ann2, image_ids, txt2img

def get_answer_row_col():
    data = read_json(Qwen2VL_answer_path)
    print(len(data))
    assert len(data) == len(ann1) * question_num * top_cnt
    answer_row_col, error_list = [], []
    error_data = read_json(question_error_path)
    print(f'len(error_data) = {len(error_data)}')
    for item in error_data:
        error_list.append(item["idx"])
    for i in tqdm(range(0, len(data), question_num)):
        row, col = data[i]["ori_cap_idx"], data[i][f"topk_{config['type']}_idx"]
        all_answer = ""
        for j in range(i, i + question_num):
            all_answer += data[j]["answer"]
        all_answer = all_answer.replace("Uncertain", "")
        if row in error_list:
            all_answer = ""
        answer_row_col.append({
            "row":row,
            "col":col,
            "all_answer":all_answer,
        })
    print(len(answer_row_col))
    assert len(answer_row_col) == len(data) // question_num
    return answer_row_col

# %%
VISA_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = load_yaml()
top_cnt, question_num = 20, 3
query_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}", f"{config['query']}")
question_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                             f"{config['question']}")
question_error_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                                   f"{config['question_error']}")
recap_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}", f"{config['recap']}")
pre_answer_index_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                                     f"{config['pre_answer_index']}")
Qwen2VL_answer_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                                   f"{config['Qwen2VL_answer']}")
all_scores_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}", f"scores.pkl")
sim_base_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                             f"{config['sim_base']}")
sim_text_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                             f"{config['sim_text']}")
raw_dataset_path = os.path.join(VISA_path, "data", "raw", f"{config['dataset']}")
ann1, ann2, image_ids, txt2img = get_retrieval_element()
