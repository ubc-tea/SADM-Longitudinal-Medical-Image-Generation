# SADM: Sequence-Aware Diffusion Model
The official implementation of our IPMI 2023 paper "**[SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation](https://arxiv.org/abs/2212.08228)**"

[Jee Seok Yoon*](https://www.jsyoon.kr/), Chenghao Zhang, [Heung-Il Suk](https://milab.korea.ac.kr/), [Jia Guo](https://mr.research.columbia.edu/content/jia-guo), [Xiaoxiao Li](https://tea.ece.ubc.ca/)

## Abstract
![framework](https://user-images.githubusercontent.com/5194237/219783819-92de9233-12bf-4100-9efe-4976cb408c81.svg)
Human organs constantly undergo anatomical changes due to a complex mix of short-term (e.g., heartbeat) and long-term (e.g., aging) factors. Evidently, prior knowledge of these factors will be beneficial when modeling their future state, i.e., via image generation. However, most of the medical image generation tasks only rely on the input from a single image, thus ignoring the sequential dependency even when longitudinal data is available. Sequence-aware deep generative models, where model input is a sequence of ordered and timestamped images, are still underexplored in the medical imaging domain that is featured by several unique challenges: 1) Sequences with various lengths; 2) Missing data or frame, and 3) High dimensionality. To this end, we propose a sequence-aware diffusion model (SADM) for the generation of longitudinal medical images. Recently, diffusion models have shown promising results in high-fidelity image generation. Our method extends this new technique by introducing a sequence-aware transformer as the conditional module in a diffusion model. The novel design enables learning longitudinal dependency even with missing data during training and allows autoregressive generation of a sequence of images during inference. Our extensive experiments on 3D longitudinal medical images demonstrate the effectiveness of SADM compared with baselines and alternative methods.


## Usage
Currently, this repository only contains the minimal and enssential code for training and validation (meaning it won't produce the same outcomes as detailed in the paper). We are in the process of cleaning up our internal code for the general public (e.g., removing multi-GPU coding designed for our cluster). In the mean time, changing [``n_epoch=300``](https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation/blob/main/SADM.py#L17) should get a similar result as the paper. We applogize for the inconvience, and we will update the code as soon as possible. Please stay tuned!
### Setup
Make sure to install all packages in ``requirements.txt``. If needed, change ``DATA_DIR`` and ``RESULT_DIR`` in SADM.py and ACDC_prepare.py.
### Training and Validation
1. Download and prepare dataset: ``python ACDC_prepare.py``
1. Train SADM: ``python SADM.py``

ViVit code has been modified from [rishikksh20's code](https://github.com/rishikksh20/ViViT-pytorch), and DDPM has been modified from [TeaPearce's code](https://github.com/TeaPearce/Conditional_Diffusion_MNIST).


## Citation

````
@InProceedings{yoon2022sadm,
  title={SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation},
  author={Yoon, Jee Seok and Zhang, Chenghao and Suk, Heung-Il and Guo, Jia and Li, Xiaoxiao},
  booktitle={Information Processing in Medical Imaging},
  year={2023}
}
````
