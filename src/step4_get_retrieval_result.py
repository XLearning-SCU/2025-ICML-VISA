import pickle
from utils import *

# %%
def main():
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
    all_scores = []
    for i in range(config['gemma2_cnt_parts']):
        score_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                                  f"scores_part[{i}-{config['gemma2_cnt_parts']}].pkl")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)
        all_scores.extend(scores)
    sim_text = np.zeros((len(ann1), len(ann2)))
    idx = 0
    answer_row_col = get_answer_row_col()
    for item in tqdm(answer_row_col):
        i, j = item["row"], item["col"]
        sim_text[i, j] = all_scores[idx]
        idx += 1
    np.savetxt(sim_text_path, sim_text, fmt = '%.10f', delimiter = ' ')
    # %%
    sim_base = np.loadtxt(sim_base_path)
    sim_text = np.loadtxt(sim_text_path)
    top_indices = np.argsort(-sim_base, axis = 1)[:, :top_cnt]
    together_score = norm(sim_base) + norm(sim_text)
    print(f"The text-to-{config['type']} retrieval result of {config['dataset']}({config['base_model']}) is (R@1|R@5|R@10):", end = " ")
    out_to_typora_t2i(together_score, txt2img)

if __name__ == '__main__':
    main()
