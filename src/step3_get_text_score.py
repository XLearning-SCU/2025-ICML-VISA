import os.path
import sys

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils import *

# %%
def main():
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
    if not os.path.exists(Qwen2VL_answer_path):
        data = []
        for i in range(config["Qwen2VL_cnt_parts"]):
            Qwen2VL_answer_part_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                                                    f"Qwen2VL_answer_part[{i}-{config['Qwen2VL_cnt_parts']}].json")
            data.extend(read_json(Qwen2VL_answer_part_path))
        print(f'len(data) = {len(data)}')
        write_json(Qwen2VL_answer_path, data)
    # %%
    answer_row_col = get_answer_row_col()
    span = len(answer_row_col) // config['gemma2_cnt_parts']
    begin = span * config['gemma2_current_part']
    end = begin + span
    if config['gemma2_current_part'] == config['gemma2_cnt_parts'] - 1:
        end = len(answer_row_col)
    print(f'span = {span}, begin = {begin}, end = {end}, len(answer_row_col) = {len(answer_row_col)}')
    # %%
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config['gemma2_current_gpu']}"
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['gemma2_model_path'],
                                                           trust_remote_code = True)
    tokenizer.padding_side = 'right'
    model = transformers.AutoModelForCausalLM.from_pretrained(config['gemma2_model_path'],
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
    batch_size = config['gemma2_batch_size']
    dataset = PairsDataset(pairs)
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = False,
                            collate_fn = collate_fn,
                            num_workers = 4,
                            pin_memory = True)
    # %%
    print(f'-' * 100)
    scores = []
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
    score_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                              f"scores_part[{config['gemma2_current_part']}-{config['gemma2_cnt_parts']}].pkl")
    with open(score_path, "wb") as f:
        pickle.dump(scores, f)

if __name__ == '__main__':
    main()
