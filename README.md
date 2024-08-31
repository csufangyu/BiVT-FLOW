# BiVT-FLOW

This is the official implementation of the paper
## [Bidirectional visual-tactile cross-modal generation using latent feature space flow model](https://doi.org/10.1016/j.neunet.2023.12.042) 

## Installation

```
conda env create -f environment.yaml
conda activate bivt
```

The touch2go Datasets can be find at [Touch2go](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B)

## Train

Our training is divided into two stages, the first stage is to train VAE model, and the second stage is to train flow model.

For the VAE, we train the vision VAE and the tactile VAE. 

Vision VAE:

```
python main.py --base configs/vision_encoderkl.yaml -t True --gpus 0
```

tactile VAE:

```
python main.py --base configs/gel_encoderkl.yaml -t True --gpus 0
```
In order to speed up the training process, we can use the trained VAE model to compress the corresponding image into the latent space.
```
cd flow
python extract_latent.py
```

Then we can train flow model.
```
python bivtflow.py
```

## Acknowledgement

Our code is generally built upon: [taming-transformers](https://github.com/CompVis/taming-transformers/tree/master),[latent-diffusion](https://github.com/CompVis/latent-diffusion),[RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)

## BibTeX
```
@article{fang2024bidirectional,
  title={Bidirectional visual-tactile cross-modal generation using latent feature space flow model},
  author={Fang, Yu and Zhang, Xuehe and Xu, Wenqiang and Liu, Gangfeng and Zhao, Jie},
  journal={Neural Networks},
  volume={172},
  pages={106088},
  year={2024},
  publisher={Elsevier}
}
```