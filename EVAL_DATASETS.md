# Eval datasets list

**Short-context image dataset：**

- COCO
- Flickr30K

**Short-context video dataset：**

- MSRVTT
- LSMDC
- DiDeMo
- MSVD

**Long-context dataset：**

- DCI
- IIW
- Urban1K
- ShareGPT4v

# Short-context image dataset

- COCO：Download the dataset from the official website: [https://cocodataset.org/#download](https://cocodataset.org/#download).  
  After downloading, extract the **test set** and place the files under the `raw/COCO/` directory.  
  **Repeat the same steps for all remaining datasets.**
- Flickr30K：Download it from Kaggle: [https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). 

# Short-context video dataset

For MSR-VTT, DiDeMo, and MSVD video-text datasets, please refer to the following resource for detailed download [InternVideo2 DATASET.md](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/DATASET.md#video-text-retrieval).

Due to copyright restrictions, we do not provide a direct download link for the LSMDC dataset. Please obtain the dataset through the official channels if you are authorized to access it.

# Long-context dataset

For the DCI, IIW, and ShareGPT4V datasets used in long-text-to-image retrieval evaluation, please refer to: [LoTLIP: EVAL_DATASETS.md](https://github.com/wuw2019/LoTLIP/blob/main/EVAL_DATASETS.md#data-preparation-for-long-text-image-retrieval)

The Urban1K dataset can be accessed directly from Hugging Face: [Urban1K on Hugging Face](https://huggingface.co/datasets/BeichenZhang/Urban1k)
