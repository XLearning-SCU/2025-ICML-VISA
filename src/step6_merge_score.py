import pickle
import sys
import argparse
import os

# %%
parser = argparse.ArgumentParser(description = "这是一个参数解析示例")
parser.add_argument("--cnt_parts", type = int, help = "part的数量")
args = parser.parse_args()
assert args.cnt_parts is not None
VISA_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

all_scores = []
for i in range(args.cnt_parts):
    score_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                              f"scores_part{i}-{args.cnt_parts}.pkl")
    with open(score_path, "rb") as f:
        scores = pickle.load(f)
    all_scores.extend(scores)
print(f'len(all_scores) = {len(all_scores)}')
score_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                          f"scores.pkl")
with open(score_path, "wb") as f:
    pickle.dump(all_scores, f)
sys.exit()
