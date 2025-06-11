import os
import sys
import torch
import json
from tqdm import tqdm
from collections import defaultdict
import time
from datetime import datetime
import argparse

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

# %%
parser = argparse.ArgumentParser(description = "这是一个参数解析示例")
parser.add_argument("--cnt_parts", type = int, help = "part的数量")
parser.add_argument("--current_part", type = int, help = "当前part的下标")
parser.add_argument("--current_gpu", type = str, help = "当前使用的gpu")
args = parser.parse_args()
assert args.cnt_parts is not None
assert args.current_part is not None
assert args.current_gpu is not None
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
def get_answer(messages):
    texts = [
        processor.apply_chat_template(msg, tokenize = False, add_generation_prompt = True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text = texts,
            images = image_inputs,
            videos = video_inputs,
            padding = True,
            return_tensors = "pt",
    )
    inputs = inputs.to("cuda")

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens = 1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens = True, clean_up_tokenization_spaces = False
    )
    return output_texts

# %%
question_num = 3
pre_answer_index_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "pre_answer_index.json")
pre_answer_index = read_json(pre_answer_index_path)
key_phrases_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Flickr30K_question.json")
key_phrases = read_json(key_phrases_path)
print(f'len(pre_answer_index) = {len(pre_answer_index)}')
print(f'len(key_phrases) = {len(key_phrases)}')
assert len(pre_answer_index) == len(key_phrases) * question_num * 20
messages = []
for idx, item in enumerate(pre_answer_index):
    image_path = os.path.join(VISA_path, "data", "raw", "flickr30k-images", item["topk_image_id"])
    question = item["question"]
    key_p = key_phrases[idx // (question_num * 20)]["key_phrases"]
    question = (
        f"You need to answer the following question about the given image in English: {question}\n. "
        f"Avoid giving answers that are just 'Yes', 'No', or a single word. Each response should provide a complete sentence.\n")
    question += "If you cannot determine the answer or there are no objects that are asked by the question, just answer a single word 'Uncertain'.\n"
    question += f"Note that if the given image is unrelated or only loosely related to these phrases:{key_p}, just response a single word 'Uncertain'."
    message = [
        {
            "role":"user",
            "content":[
                {"type":"image", "image":image_path},
                # For video, we only set the fps to 3.
                # {
                #     "type":"video",
                #     "video":video_path,
                #     "max_pixels":360 * 420,
                #     "fps":3,
                # },
                {"type":"text", "text":question},
            ],
        }
    ]
    messages.append(message)
print(f'len(messages) = {len(messages)}')
# %%
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.current_gpu}"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
        device_map = "auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# %%
span = len(messages) // args.cnt_parts
begin = span * args.current_part
end = begin + span
if args.current_part == args.cnt_parts - 1:
    end = len(messages)
content, last = [], None
Qwen2VL_answer_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                                   f"Qwen2VL_answer_part{args.current_part}-{args.cnt_parts}.json")
print(f"len(content) = {len(content)}")
if not os.path.exists(Qwen2VL_answer_path):
    start_id, content = begin, []
else:
    content = read_json(Qwen2VL_answer_path)
    start_id = content[-1]["idx"] + 1
assert start_id is not None
# %%
batch_size = 10
start_time = time.time()
for b in tqdm(range(start_id, end, batch_size)):
    end_idx = min(b + batch_size, end)
    batch_messages = messages[b:end_idx]
    batch_idx = range(b, end_idx)
    responses = get_answer(batch_messages)
    for idx, response in zip(batch_idx, responses):
        temp = pre_answer_index[idx]
        content.append({
            "idx":idx,
            "ori_cap_idx":temp["ori_cap_idx"],
            "ori_cap_to_image_id":temp["ori_cap_to_image_id"],
            "ori_cap":temp["ori_cap"],
            "topk_image_idx":temp["topk_image_idx"],
            "topk_image_id":temp["topk_image_id"],
            "question_idx":temp["question_idx"],
            "question":temp["question"],
            "answer":response,
        })
    if (b - start_id) % 20 == 0 or b > end - 100:
        write_json(Qwen2VL_answer_path, content)
        last = b
    print(f'start_id = {start_id}, Qwen2VL_answer_path = {Qwen2VL_answer_path}, len(content) = {len(content)}')
    print(f'args = {args}')
    print(f'span = {span}, begin = {begin}, end = {end}, end - begin = {end - begin}, len(messages) = {len(messages)}')
    print(f'b = {b}, end_idx = {end_idx}, last = {last}')
    output_time(start_time)
    print('-' * 100)
sys.exit()
