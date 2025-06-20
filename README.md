# Visual Abstraction: A Plug-and-Play Approach for Text-Visual Retrieval

- ##### **Authors:** [Guofeng Ding](https://scholar.google.com.hk/citations?user=oywAwDwAAAAJ&hl=zh-CN&oi=ao), [Yiding Lu](https://object907.github.io/), [Peng Hu](https://penghu-cs.github.io/), [Mouxing Yang](https://mouxingyang.github.io/), [Yijie Lin](https://lin-yijie.github.io/), [Xi Peng](https://pengxi.me/)<br>

- ##### **Resources**: [Paper](http://pengxi.me/wp-content/uploads/2025/05/2025ICML.pdf)

##### **Accepted by ICML 2025**

## News

- [2025/05/01] VISA is accepted by ICML 2025

## Highlights

- Natural language exhibits higher semantic density compared to visual signals.

<img src="figures/VISA.png" alt="paper" style="zoom: 50%;" />

- Proposes abstracting visual signals into natural language and aligning modalities via a question-answering mechanism, effectively resolving cross-modal inconsistencies in semantic density and granularity, and significantly improving retrieval performance.

<img src="figures/example.png" alt="paper" style="zoom: 50%;" />

## Install

First, 

```
conda create -n VISA python=3.10
conda activate VISA

pip install -r requirements.txt
```

Then, download the `.whl` files for [**FlashAttention**](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [**FlashInfer**](https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06) , then install them using pip:

```
pip install flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl
```

## Datasets

For usage instructions of all datasets, please refer to [EVAL_DATASETS.md](EVAL_DATASETS.md).

## Retrieval

> Use the Flickr30K (EVA-CLIP-based) dataset as an example

-  Launch the SGLang inference server with 4 GPUs, loading the Qwen2.5-32B-Instruct model and exposing it via HTTP

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 \
$$/path/to/anaconda3/envs/VISA/bin/python$$ -m sglang.launch_server \
  --model-path $$/path/to/Qwen2.5-32B-Instruct$$ \
  --tp 4 \
  --enable-p2p-check \
  --mem-fraction-static 0.8 \
  --host "0.0.0.0" \
  --disable-cuda-graph \
  --port 12345
```

- retrieval

```python
bash run.sh src/step1_generate_question.py -- \
src/step2_answer_question.py -- \
src/step3_get_text_score.py -- \
src/step4_get_retrieval_result.py
```

### Multi-GPU Inference

For step2, the inference process can be split into multiple parts and assigned to different GPUs for parallel execution.

```python
python src/step2_answer_question.py
```

**Key parameters (set in Flickr30K(EVA-CLIP).yaml):**

- `Qwen2VL_cnt_parts`: total number of parts to divide the dataset into (e.g., 4)
- `Qwen2VL_current_part`: the index of the current part to process (starting from 0)
- `Qwen2VL_current_gpu`: the GPU ID to use for the current part

This setup allows you to run multiple processes in parallel, each handling a different slice of the dataset on a different GPU.

You can run **step 2 independently** from other steps. **Step 3 works in the same way**, using its own parameters:

`gemma2_cnt_parts`, `gemma2_current_part`, and `gemma2_current_gpu`.

### Another dataset

To evaluate on a different dataset, open `config/EVAL_DATASET.yaml` and uncomment the line corresponding to the dataset you want to use by setting:

```
EVAL_DATASET_name: "Flickr30K(EVA-CLIP)"
```

Only one dataset should be active at a time.

### Intermediate files

All intermediate files required for this project are available at the following Hugging Face link:

👉 [https://huggingface.co/datasets/XLearning-SCU/VISA](https://huggingface.co/datasets/XLearning-SCU/VISA)

You can download them directly and place them in the appropriate `data` directory.

## Evaluation Results

> The retrieval results are presented in the format of **(R@1 | R@5 | R@10)**.  * indicates results are re-evaluated using official checkpoints from HuggingFace.

<img src="figures/result.png" alt="paper" style="zoom: 50%;" />

## Bibtex

If you find this repository helpful, please consider giving it a ⭐️ and citing our work — your support is greatly appreciated!

```
@inproceedings{ding2025visual,
  title={Visual Abstraction: A Plug-and-Play Approach for Text-Visual Retrieval},
  author={Ding, Guofeng and Lu, Yiding and Hu, Peng and Yang, Mouxing and Lin, Yijie and Peng, Xi},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
}
```

## Acknowledgements

We would like to express our gratitude to [**SigLIP**](https://arxiv.org/abs/2303.15343), [**EVA-CLIP**](https://arxiv.org/abs/2402.04252), [**InterVideo2**](https://arxiv.org/abs/2403.15377), and [**LoTLIP**](https://arxiv.org/abs/2410.05249) for their excellent work, as well as to [**LLaVA**](https://huggingface.co/liuhaotian/llava-v1.6-34b), [**Qwen**](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), and [**BGE**](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) for providing powerful foundation models.

