import os
import sys

import numpy as np
import time
import json
from collections import defaultdict
import torch
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader

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
    R50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    R100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    return R1, R5, R10, R50, R100

def last_logit_pool(logits: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim = 1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim = 0)

class PairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, passage = self.pairs[idx]
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors = None,
                                 add_special_tokens = False,
                                 max_length = max_length * 3 // 4,
                                 truncation = True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors = None,
                                   add_special_tokens = False,
                                   max_length = max_length,
                                   truncation = True)
        item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation = 'only_second',
                max_length = max_length,
                padding = False,
                return_attention_mask = False,
                return_token_type_ids = False,
                add_special_tokens = False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        return query_inputs, item

def collate_fn(batch):
    inputs = []
    query_lengths = []
    prompt_lengths = []
    for query_inputs, item in batch:
        inputs.append(item)
        query_lengths.append(len([tokenizer.bos_token_id] + query_inputs['input_ids'] + sep_inputs))
        prompt_lengths.append(len(sep_inputs + prompt_inputs))

    return tokenizer.pad(
            inputs,
            padding = True,
            max_length = max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of = 8,
            return_tensors = 'pt',
    ), query_lengths, prompt_lengths

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
import argparse

parser = argparse.ArgumentParser(description = "这是一个参数解析示例")
parser.add_argument("--cnt_parts", type = int, help = "part的数量")
parser.add_argument("--current_part", type = int, help = "当前part的下标")
parser.add_argument("--current_gpu", type = str, help = "当前使用的gpu")
args = parser.parse_args()
assert args.cnt_parts is not None
assert args.current_part is not None
assert args.current_gpu is not None
assert args.current_part < args.cnt_parts
# assert args.current_gpu < 8
print(f'args = {args}')
span = len(answer_row_col) // args.cnt_parts
begin = span * args.current_part
end = begin + span
if args.current_part == args.cnt_parts - 1:
    end = len(answer_row_col)
print(f'span = {span}, begin = {begin}, end = {end}, len(answer_row_col) = {len(answer_row_col)}')
# %%
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.current_gpu}"
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2.5-gemma2-lightweight',
                                                       trust_remote_code = True)
tokenizer.padding_side = 'right'
model = transformers.AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2.5-gemma2-lightweight',
                                                          trust_remote_code = True,
                                                          torch_dtype = torch.float16)
model = model.to("cuda")
model.eval()
max_length = 1024
prompt = "Predict whether passage B contains an answer to query A."
sep = "\n"
prompt_inputs = tokenizer(prompt,
                          return_tensors = None,
                          add_special_tokens = False)['input_ids']
sep_inputs = tokenizer(sep,
                       return_tensors = None,
                       add_special_tokens = False)['input_ids']
# %%
pairs = []
for item in tqdm(answer_row_col[begin:end]):
    i, j = item["row"], item["col"]
    pairs.append([ann1[i], item["all_answer"] + ann2[j]])
batch_size = 10
dataset = PairsDataset(pairs)
dataloader = DataLoader(dataset,
                        batch_size = batch_size,
                        shuffle = False,
                        collate_fn = collate_fn,
                        num_workers = 4,
                        pin_memory = True)
# %%
print(f'-' * 100)
start_time = time.time()
scores = []
sim_t2i = np.zeros((len(ann1), len(ann2)))
with torch.no_grad():
    for batch in tqdm(dataloader):
        inputs, query_lengths, prompt_lengths = batch
        inputs = inputs.to(model.device)
        outputs = model(**inputs,
                        return_dict = True,
                        cutoff_layers = [28],
                        compress_ratio = 2,
                        compress_layer = [24, 40],
                        query_lengths = query_lengths,
                        prompt_lengths = prompt_lengths)
        score = []
        for k in range(len(outputs.logits)):
            logits = last_logit_pool(outputs.logits[k], outputs.attention_masks[k])
            score.append(logits.cpu().float().tolist())
        scores.extend(score[0])
score_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                          f"scores_part{args.current_part}-{args.cnt_parts}.pkl")
with open(score_path, "wb") as f:
    pickle.dump(scores, f)
sys.exit()
