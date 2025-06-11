import sys
import os
import numpy as np
import time
from datetime import datetime
import json
from collections import defaultdict

# %%
def output_time(start_time):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"{current_time} Program runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")

def read_json(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        return json.load(file)

def write_json(path, content):
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(content, f, ensure_ascii = False, indent = 4)

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

# %%
print(f'-' * 50)
print(f'\033[31mannflickr\033[0m')
ann1 = []
short_dic = defaultdict(list)
VISA_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'VISA_path = {VISA_path}')
query_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Flickr30K_query.json")
short_data = read_json(query_path)
for item in short_data:
    short_dic[item['image_id']].append(item["captions"][0])
    ann1.append(item["captions"][0])

ann2, image_ids = [], []
recap_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Flickr30K_recap.json")
long_data = read_json(recap_path)
for item in long_data:
    ann2.append(item["captions"][0])
    image_ids.append(item["image_id"])
print(f'len(ann1) = {len(ann1)}')
print(f'len(ann2) = {len(ann2)}')
print(f'len(image_ids) = {len(image_ids)}')

txt2img, img2txt = {}, {}

txt_id = 0
for img_id, (key, value) in enumerate(short_dic.items()):
    img2txt[img_id] = []
    for i, caption in enumerate(value):
        img2txt[img_id].append(txt_id)
        txt2img[txt_id] = img_id
        txt_id += 1
print(f'-' * 50)

# %%
def norm(matrix):
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        top_vals = matrix[i, top_indices[i]]
        min_val = top_vals.min()
        max_val = top_vals.max()
        if max_val > min_val:
            norm_vals = (top_vals - min_val) / (max_val - min_val)
        else:
            norm_vals = np.zeros_like(top_vals)  # 如果 max == min，则归一化后所有值都为 0
        normalized_matrix[i, top_indices[i]] = norm_vals
    return normalized_matrix

# %%
base_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "sim_Flickr30K(EVA-CLIP).txt")
sim_base = np.loadtxt(base_path)
txt_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                        f"sim_text.txt")
sim_text = np.loadtxt(txt_path)
top_cnt = 20
top_indices = np.argsort(-sim_base, axis = 1)[:, :top_cnt]
together_score = norm(sim_base) + norm(sim_text)
print(f'The retrieval result of Flickr30K(EVA-CLIP) is:', end = " ")
out_to_typora_t2i(together_score, txt2img)
# %%
sys.exit()
