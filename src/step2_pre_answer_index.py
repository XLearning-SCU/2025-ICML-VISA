import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

def read_json(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        return json.load(file)

def write_json(path, content):
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(content, f, ensure_ascii = False, indent = 4)

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
question_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Flickr30K_question.json")
all_questions = read_json(question_path)
print(f'len(all_questions) = {len(all_questions)}')
base_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "sim_Flickr30K(EVA-CLIP).txt")
sim_base = np.loadtxt(base_path)
top_cnt = 20
top_indices = np.argsort(-sim_base, axis = 1)[:, :top_cnt]
print(f'-' * 100)
start_time = time.time()
content = []
idx = 0
for i in range(len(ann1)):
    for j in top_indices[i]:
        for question_idx, question in enumerate(all_questions[i]["questions"]):
            content.append({
                "idx":idx,
                "ori_cap_idx":i,
                "ori_cap_to_image_id":image_ids[txt2img[i]],
                "ori_cap":ann1[i],
                "topk_image_idx":int(j),
                "topk_image_id":image_ids[j],
                "question_idx":question_idx,
                "question":question
            })
            idx += 1
print(f'len(content) = {len(content)}')
pre_answer_index_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "pre_answer_index.json")
write_json(pre_answer_index_path, content)
sys.exit()
