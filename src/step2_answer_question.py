import sys
import torch
from utils import *

# %%
def main():
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
    pre_answer_index = read_json(pre_answer_index_path)
    key_phrases = read_json(question_path)
    print(f'len(pre_answer_index) = {len(pre_answer_index)}')
    print(f'len(key_phrases) = {len(key_phrases)}')
    assert len(pre_answer_index) == len(key_phrases) * question_num * top_cnt
    messages = []
    for idx, item in enumerate(pre_answer_index):
        if config["type"] == "image":
            image_path = os.path.join(raw_dataset_path, item["topk_image_id"])
            temp_content = {"type":"image", "image":image_path}
        elif config["type"] == "video":
            video_path = os.path.join(raw_dataset_path, item["topk_video_id"])
            temp_content = {"type":"video", "video":video_path, "max_pixels":360 * 420, "fps":3}
        else:
            assert False
        question = item["question"]
        key_p = key_phrases[idx // (question_num * top_cnt)]["key_phrases"]
        question = (
            f"You need to answer the following question about the given {config['type']} in English: {question}\n. "
            f"Avoid giving answers that are just 'Yes', 'No', or a single word. Each response should provide a complete sentence.\n")
        question += "If you cannot determine the answer or there are no objects that are asked by the question, just answer a single word 'Uncertain'.\n"
        question += f"Note that if the given {config['type']} is unrelated or only loosely related to these phrases:{key_p}, just response a single word 'Uncertain'."
        message = [
            {
                "role":"user",
                "content":[
                    temp_content,
                    {"type":"text", "text":question},
                ],
            }
        ]
        messages.append(message)
    print(f'len(messages) = {len(messages)}')
    # %%
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config['Qwen2VL_current_gpu']}"
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            config['Qwen2VL_model_path'],
            torch_dtype = torch.bfloat16,
            attn_implementation = "flash_attention_2",
            device_map = "auto",
    )
    processor = AutoProcessor.from_pretrained(config['Qwen2VL_model_path'])
    # %%
    span = len(messages) // config['Qwen2VL_cnt_parts']
    begin = span * config['Qwen2VL_current_part']
    end = begin + span
    if config["Qwen2VL_current_part"] == config["Qwen2VL_cnt_parts"] - 1:
        end = len(messages)
    content, last = [], None
    Qwen2VL_answer_part_path = os.path.join(VISA_path, "data", "processed", f"{config['processed_file_path']}",
                                            f"Qwen2VL_answer_part[{config['Qwen2VL_current_part']}-{config['Qwen2VL_cnt_parts']}].json")
    print(f"len(content) = {len(content)}")
    if not os.path.exists(Qwen2VL_answer_part_path):
        start_id, content = begin, []
    else:
        content = read_json(Qwen2VL_answer_part_path)
        start_id = content[-1]["idx"] + 1
    assert start_id is not None
    # %%
    batch_size = config['Qwen2VL_batch_size']
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
                f"ori_cap_to_{config['type']}_id":temp[f"ori_cap_to_{config['type']}_id"],
                "ori_cap":temp["ori_cap"],
                f"topk_{config['type']}_idx":temp[f"topk_{config['type']}_idx"],
                f"topk_{config['type']}_id":temp[f"topk_{config['type']}_id"],
                "question_idx":temp["question_idx"],
                "question":temp["question"],
                "answer":response,
            })
        if (b - start_id) % 20 == 0 or b > end - 100:
            write_json(Qwen2VL_answer_part_path, content)
            last = b
        print(
                f'start_id = {start_id}, Qwen2VL_answer_part_path = {Qwen2VL_answer_part_path}, len(content) = {len(content)}')
        print(
                f'span = {span}, begin = {begin}, end = {end}, end - begin = {end - begin}, len(messages) = {len(messages)}')
        print(f'b = {b}, end_idx = {end_idx}, last = {last}')
        output_time(start_time)
        print('-' * 100)

if __name__ == '__main__':
    main()
