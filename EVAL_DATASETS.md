# Eval datasets list:

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

- COCO：Download it from the official website:  [https://cocodataset.org/#download](https://cocodataset.org/#download). After downloading, place the files under the **`raw/COCO/`** directory.

- Flickr30K：Download it from Kaggle: [https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).

# Short-context video dataset

For MSR-VTT, DiDeMo, and MSVD video-text datasets, please refer to the following resource for detailed download and preprocessing instructions: [InternVideo2 DATASET.md](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/DATASET.md#video-text-retrieval).

Follow the guidelines there to obtain and organize the datasets accordingly.  
After downloading, place the data under the corresponding subdirectories in `raw/` (e.g., `raw/MSRVTT/`, `raw/DiDeMo/`, `raw/MSVD/`).

Due to copyright restrictions, we do not provide a direct download link for the LSMDC dataset.  
Please obtain the dataset through the official channels if you are authorized to access it.

Once obtained, place the files under the `raw/LSMDC/` directory.

# Long-context dataset

For the DCI, IIW, and ShareGPT4V datasets used in long-text-to-image retrieval evaluation, please refer to the detailed preparation instructions provided here: [LoTLIP: EVAL_DATASETS.md - Data Preparation](https://github.com/wuw2019/LoTLIP/blob/main/EVAL_DATASETS.md#data-preparation-for-long-text-image-retrieval)

Follow the steps in the linked documentation to download and format the datasets appropriately.  
Once processed, organize the data under the corresponding subdirectories in `raw/` (e.g., `raw/DCI/`, `raw/IIW/`, `raw/ShareGPT4V/`).

The Urban1K dataset can be accessed directly from Hugging Face: [Urban1K on Hugging Face](https://huggingface.co/datasets/BeichenZhang/Urban1k)

After downloading, place the dataset in the `raw/Urban1K/` directory.