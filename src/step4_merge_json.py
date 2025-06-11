import json
import os
import argparse
def read_json(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        return json.load(file)

def write_json(path, content):
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(content, f, ensure_ascii = False, indent = 4)

# %%
parser = argparse.ArgumentParser(description = "这是一个参数解析示例")
parser.add_argument("--cnt_parts", type = int, help = "part的数量")
args = parser.parse_args()
assert args.cnt_parts is not None
VISA_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'VISA_path = {VISA_path}')
data = []
for i in range(args.cnt_parts):
    answer_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", f"Qwen2VL_answer_part{i}-{args.cnt_parts}.json")
    data.extend(read_json(answer_path))
print(len(data))
write_json(os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Qwen2VL_answer.json"), data)
