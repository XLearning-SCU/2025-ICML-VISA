import json
import pickle
import time
from collections import defaultdict
import os
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# %%
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
question_num = 3
Qwen2VL_answer_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Qwen2VL_answer.json")
data = read_json(Qwen2VL_answer_path)
print(len(data))
assert len(data) == len(ann1) * question_num * 20
answer_row_col = []
error_list = []
error_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Flickr30K_question_error.json")
error_data = read_json(error_path)
print(f'len(error_data) = {len(error_data)}')
for item in error_data:
    error_list.append(item["idx"])
for i in tqdm(range(0, len(data), question_num)):
    row, col = data[i]["ori_cap_idx"], data[i]["topk_image_idx"]
    all_answer = ""
    for j in range(i, i + question_num):
        all_answer += data[j]["answer"]
    all_answer = all_answer.replace("Uncertain", "")
    if i // (question_num * 20) in error_list:
        all_answer = ""
    answer_row_col.append({
        "row":row,
        "col":col,
        "all_answer":all_answer,
    })
print(len(answer_row_col))
assert len(answer_row_col) == len(data) // question_num

# %%
score_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                          f"scores.pkl")
with open(score_path, "rb") as f:
    scores = pickle.load(f)
print(f'-' * 100)
start_time = time.time()
sim_text = np.zeros((len(ann1), len(ann2)))
idx = 0
for item in tqdm(answer_row_col):
    i, j = item["row"], item["col"]
    sim_text[i, j] = scores[idx]
    idx += 1

out_to_typora_t2i(sim_text, txt2img)
print(f'sim_text.shape = {sim_text.shape}')
txt_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                        f"sim_text.txt")
np.savetxt(txt_path, sim_text, fmt = '%.10f', delimiter = ' ')
