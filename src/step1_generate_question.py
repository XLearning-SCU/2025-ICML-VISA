import os
import re
from datetime import datetime

from tqdm import tqdm

from llava_labeler_api import *

def output_time(start_time):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"{current_time} Program runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")

def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def read_json(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        return json.load(file)

def write_json(path, content):
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(content, f, ensure_ascii = False, indent = 4)

def load_file_and_processor(anno_file):
    annos = json.load(open(anno_file, 'r'))
    n_sample = len(annos)
    print(f'n_sample: {n_sample}')
    batch_processor = OpenAIBatchProcessor("http://192.168.49.59:12345/v1",
                                           api_key = "EMPTY")
    return annos, batch_processor, n_sample

def describe_pipline(processor: OpenAIBatchProcessor, img_dir, annos, indices) -> list:
    end_point = "/v1/chat/completions"
    fine_jsonl_file = []
    system_prompt = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""
    for idx, anno in zip(indices, annos):
        messages = f"""You are observing a image. A single sentence is provided to describe what you are seeing in the image. Based on this sentence, perform the following tasks:
1. Extract Key Phrases: Identify and list key phrases from the sentence that represent the main elements of the scene. Focus on capturing: object types, object attributes, object actions, object locations, interactions between objects or other dynamic elements
2. Generate Three Questions: Based on the extracted key phrases, create three natural language questions that a visual AI assistant might ask about the image. The questions should:
   2.1 Be specific and focused on the visual details described in the sentence.
   2.2 Ensure that the answer to each question can be determined confidently:
     - Either the content is present and can be answered confidently from the image.
     - Or the content is absent, and its non-presence can be confidently verified.
   2.3 Only ask questions about quantities (e.g., "How many...") if the description explicitly mentions a number or quantity (e.g., "three," "several").
   2.4 Avoid directly repeating or closely paraphrasing most of the sentence.
   2.5 Ensure the answers to the questions can be provided using the extracted key phrases.

Output Requirements:
Key phrases: Provide a numbered list of key phrases extracted from the input sentence.
Questions: Provide a numbered list of three relevant questions based on the key phrases.

Example:
Input:
A green and yellow tennis sweater is hanging on the back of the sofa.

Output:
key phrases:
1. green and yellow
2. tennis sweater
3. a green and yellow tennis sweater
4. hanging on
5. the back of
6. sofa
7. the back of the sofa
8. hanging on the back of the sofa
questions:
1. What colors are present on the tennis sweater?
2. Where is the tennis sweater located?
3. What type of object is the sweater hanging on?

Now, perform these tasks for the following input:
{anno["captions"][0]}"""
        request = format_request(idx, system_prompt, messages, None, end_point,
                                 temperature = 0.2,
                                 max_tokens = 1024)
        fine_jsonl_file.append(request)

    input_file_path = f"batch_request.jsonl"
    with open(input_file_path, 'w') as f:
        for entry in fine_jsonl_file:
            f.write(json.dumps(entry) + '\n')
    fine_response = processor.process_batch(input_file_path, end_point, completion_window = "24h")

    descriptions = []
    for idx, fres in zip(indices, fine_response):
        caption = fres["response"]["body"]["choices"]["message"]["content"]
        caption = caption.replace("\n", " ")
        descriptions.append({"caption":caption})

    return descriptions

def description_labeler():
    VISA_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'VISA_path = {VISA_path}')
    anno_file = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP", "Flickr30K_query.json")
    annos, processor, n_sample = load_file_and_processor(anno_file)
    img_dir = ""
    Flickr30K_question_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                                           "Flickr30K_question_.json")
    Flickr30K_question_error_path = os.path.join(VISA_path, "data", "processed", "Flickr30K", "EVA-CLIP",
                                                 "Flickr30K_question_error_.json")

    start_id, content, error, error_cnt = None, None, None, None
    if not os.path.exists(Flickr30K_question_path):
        start_id, content, error, error_cnt = 0, [], [], 0
    else:
        content = read_json(Flickr30K_question_path)
        start_id = content[-1]["idx"] + 1
        if os.path.exists(Flickr30K_question_error_path):
            error = read_json(Flickr30K_question_error_path)
            error_cnt = len(error)
        else:
            error, error_cnt = [], 0
    assert start_id is not None
    assert content is not None
    assert error is not None
    assert error_cnt is not None
    print(f"start_id = {start_id}, len(content) = {len(content)}, error_cnt = {error_cnt}")

    batch_size = 100
    start_time = time.time()
    question_num = 3
    for b in tqdm(range(start_id, n_sample, batch_size)):
        end_idx = min(n_sample, b + batch_size)
        batch_idx = range(b, end_idx)
        descriptions = describe_pipline(processor, img_dir, annos[b:end_idx], batch_idx)

        for idx, data in zip(batch_idx, descriptions):
            ori_cap = annos[idx]["captions"][0]
            annos[idx].update(data)
            # ---------------------------------------------------------------------------------------------------------#
            if contains_chinese(annos[idx]['caption']):
                print(f'idx = {idx}')
                print(annos[idx]['caption'])
            assert not contains_chinese(annos[idx]['caption'])
            # ---------------------------------------------------------------------------------------------------------#
            # print(annos[idx]['caption'])
            response = annos[idx]['caption']
            key_phrases, questions = None, None
            if "questions:" in response:
                key_phrases = response.split("questions:")[0]
                assert "key phrases:" in key_phrases
                key_phrases = key_phrases.split("key phrases:")[1]
                questions = response.split("questions:")[1]
                questions = re.split(r'\d+\.\s', questions)[1:question_num + 1]

            if questions is None:
                key_phrases = ori_cap
                error.append({
                    "idx":idx,
                    "image_id":annos[idx]["image_id"],
                    "ori_cap":ori_cap,
                    "response":response
                })
                error_cnt += 1
                questions = [
                    "What is the main object or subject being focused on in the image?",
                    "Are there any actions or movements performed by the object in the image?",
                    "Are there any notable interactions between objects in the image?"
                ]
            questions = [question.split("?")[0] + "?" for question in questions]
            assert len(questions) == question_num
            # ---------------------------------------------------------------------------------------------------------#
            content.append({
                "idx":idx,
                "image_id":annos[idx]["image_id"],
                "ori_cap":ori_cap,
                "key_phrases":key_phrases,
                "questions":questions
            })
        write_json(Flickr30K_question_path, content)
        write_json(Flickr30K_question_error_path, error)
        print(f'batch_size = {batch_size}, b = {b}, error_cnt = {error_cnt}')
        print(f'Flickr30K_question_path = {Flickr30K_question_path}')
        print(f'Flickr30K_question_error_path = {Flickr30K_question_error_path}')
        output_time(start_time)
        print(f'-' * 150)

if __name__ == '__main__':
    description_labeler()
