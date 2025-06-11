# Visual Abstraction: A Plug-and-Play Approach for Text-Visual Retrieval

- ##### **Authors:** Guofeng Ding, [Yiding Lu](https://object907.github.io/), [Peng Hu](https://penghu-cs.github.io/), [Mouxing Yang](https://mouxingyang.github.io/), [Yijie Lin](https://lin-yijie.github.io/), [Xi Peng](https://pengxi.me/)<br>

- ##### **Resources**: [Paper](http://pengxi.me/wp-content/uploads/2025/05/2025ICML.pdf)

##### **Accepted by ICML 2025**

## News

- [2025/05/01] VISA is accepted by ICML 2025 

## Highlights

- Natural language exhibits higher semantic density compared to visual signals.

![paper](VISA.png)

- Proposes abstracting visual signals into natural language and aligning modalities via a question-answering mechanism, effectively resolving cross-modal inconsistencies in semantic density and granularity, and significantly improving retrieval performance.

## Install

All required environments have been packaged. You can download them from the [link](https://huggingface.co/datasets/dingguofeng/VISA) below and install them on your server to get started quickly.

```python
tar -xzvf sglang_env.tar.gz -C $$/path/to/anaconda3/envs$$

conda activate sglang
```

## Datasets

For details and usage instructions of all datasets, please refer to [EVAL_DATASETS.md](EVAL_DATASETS.md).

## Retrieval

> Use the EVA-CLIP-based Flickr30K dataset as an example

- step 0: Launch the SGLang inference server with 4 GPUs, loading the Qwen2.5-32B-Instruct model and exposing it via HTTP

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 \
$$/path/to/anaconda3/envs/sglang/bin/python$$ -m sglang.launch_server \
  --model-path $$/path/to/Qwen2.5-32B-Instruct$$ \
  --tp 4 \
  --enable-p2p-check \
  --mem-fraction-static 0.8 \
  --host "0.0.0.0" \
  --disable-cuda-graph \
  --port 12345
```

- step1: Generate 3 questions for each query

```python
python step1_generate_question.py
```

- step2: Prepare for answer generation

```python
python step2_pre_answer_index.py
```

- step3: Answer the questions corresponding to each query

```python
python step3_answer_question_Qwen2VL.py --cnt_parts 1 --current_part 0 --current_gpu 0
```

Note that the parameter cnt_parts can be greater than 1 to split the data into multiple parts

- step4: Merge all JSON files

```python
python step4_merge_json.py --cnt_parts 1
```

- step5: Get the score of the text

```python
python step5_get_rerank_result.py --cnt_parts 1 --current_part 0 --current_gpu 0
```

- step6: Merge all score files

```python
python step6_merge_score.py --cnt_parts 1 
```

- step7: Get the text similarity matrix

```python
python step7_score_to_sim.py
```

- step8: Get the final retrieval results

```python
python step8_norm_performance.py
```

## Evaluation Results

| COCO(SigLIP)     | COCO(EVA-CLIP)   |
| ---------------- | ---------------- |
| 57.1｜80.3｜86.9 | 59.4｜81.2｜87.5 |

| Flickr30K(SigLIP) | Flickr30K(EVA-CLIP) |
| ----------------- | ------------------- |
| 85.1｜97.1｜98.6  | 86.2｜97.3｜98.6    |

| MSRVTT           | LSMDC            | DiDeMo           | MSVD             |
| ---------------- | ---------------- | ---------------- | ---------------- |
| 54.6｜75.0｜82.9 | 35.0｜53.0｜61.6 | 63.6｜86.0｜89.7 | 60.1｜84.5｜89.7 |

| DCI              | IIW                | Urban1k          | ShareGPT4v       |
| ---------------- | ------------------ | ---------------- | ---------------- |
| 75.5｜89.4｜91.6 | 98.2｜100.0｜100.0 | 94.7｜99.4｜99.8 | 97.8｜99.8｜99.9 |

## Bibtex

```python
@inproceedings{ding2025visual,
  title={Visual Abstraction: A Plug-and-Play Approach for Text-Visual Retrieval},
  author={Ding, Guofeng and Lu, Yiding and Hu, Peng and Yang, Mouxing and Lin, Yijie and Peng, Xi},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
}
```

## Acknowledgements

We would like to express our gratitude to [**SigLIP**](https://arxiv.org/abs/2303.15343), [**EVA-CLIP**](https://arxiv.org/abs/2402.04252), [**InterVideo2**](https://arxiv.org/abs/2403.15377), and [**LoTLIP**](https://arxiv.org/abs/2410.05249) for their excellent work, as well as to [**LLaVA**](https://huggingface.co/liuhaotian/llava-v1.6-34b), [**Qwen**](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), and [**BGE**](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) for providing powerful foundation models.
