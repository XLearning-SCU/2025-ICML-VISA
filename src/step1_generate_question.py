import sys

from tqdm import tqdm
from llava_labeler_api import *
from utils import *
# %%
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
        messages = f"""You are observing a {config['type']}. A single sentence is provided to describe what you are seeing in the {config['type']}. Based on this sentence, perform the following tasks:
1. Extract Key Phrases: Identify and list key phrases from the sentence that represent the main elements of the scene. Focus on capturing: object types, object attributes, object actions, object locations, interactions between objects or other dynamic elements
2. Generate Three Questions: Based on the extracted key phrases, create three natural language questions that a visual AI assistant might ask about the {config['type']}. The questions should:
   2.1 Be specific and focused on the visual details described in the sentence.
   2.2 Ensure that the answer to each question can be determined confidently:
     - Either the content is present and can be answered confidently from the {config['type']}.
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
    annos, processor, n_sample = load_file_and_processor(query_path)
    img_dir = ""
    start_id, content, error, error_cnt = None, None, None, None
    if not os.path.exists(question_path):
        start_id, content, error, error_cnt = 0, [], [], 0
    else:
        content = read_json(question_path)
        start_id = content[-1]["idx"] + 1
        if os.path.exists(question_error_path):
            error = read_json(question_error_path)
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
                    f"{config['type']}_id":annos[idx][f"{config['type']}_id"],
                    "ori_cap":ori_cap,
                    "response":response
                })
                error_cnt += 1
                questions = [
                    f"What is the main object or subject being focused on in the {config['type']}?",
                    f"Are there any actions or movements performed by the object in the {config['type']}?",
                    f"Are there any notable interactions between objects in the {config['type']}?"
                ]
            questions = [question.split("?")[0] + "?" for question in questions]
            assert len(questions) == question_num
            # ---------------------------------------------------------------------------------------------------------#
            content.append({
                "idx":idx,
                f"{config['type']}_id":annos[idx][f"{config['type']}_id"],
                "ori_cap":ori_cap,
                "key_phrases":key_phrases,
                "questions":questions
            })
        write_json(question_path, content)
        write_json(question_error_path, error)
        print(f'batch_size = {batch_size}, b = {b}, error_cnt = {error_cnt}')
        output_time(start_time)
        print(f'-' * 150)
    all_questions = read_json(question_path)
    print(f'len(all_questions) = {len(all_questions)}')
    sim_base = np.loadtxt(sim_base_path)
    top_indices = np.argsort(-sim_base, axis = 1)[:, :top_cnt]
    content = []
    idx = 0
    for i in range(len(ann1)):
        for j in top_indices[i]:
            for question_idx, question in enumerate(all_questions[i]["questions"]):
                content.append({
                    "idx":idx,
                    "ori_cap_idx":i,
                    f"ori_cap_to_{config['type']}_id":image_ids[txt2img[i]],
                    "ori_cap":ann1[i],
                    f"topk_{config['type']}_idx":int(j),
                    f"topk_{config['type']}_id":image_ids[j],
                    "question_idx":question_idx,
                    "question":question
                })
                idx += 1
    print(f'len(content) = {len(content)}')
    write_json(pre_answer_index_path, content)
if __name__ == '__main__':
    description_labeler()
