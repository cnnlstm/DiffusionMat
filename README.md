# **DiffusionMat: Alpha Matting as Sequential Refinement Learning**

[paper](https://arxiv.org/pdf/2311.13535.pdf) |  [project website](https://cnnlstm.github.io/DiffusionMat/) |  [video results](https://youtu.be/b_qQvv0R3BA)

## Abstract

In this paper, we introduce DiffusionMat, a novel image matting framework that employs a diffusion model for the transition from coarse to refined alpha mattes. Diverging from conventional methods that utilize trimaps merely as loose guidance for alpha matte prediction, our approach treats image matting as a sequential refinement learning process. This process begins with the addition of noise to trimaps and iteratively denoises them using a pre-trained diffusion model, which incrementally guides the prediction towards a clean alpha matte. The key innovation of our framework is a correction module that adjusts the output at each denoising step, ensuring that the final result is consistent with the input image's structures. We also introduce the Alpha Reliability Propagation, a novel technique designed to maximize the utility of available guidance by selectively enhancing the trimap regions with confident alpha information, thus simplifying the correction task. To train the correction module, we devise specialized loss functions that target the accuracy of the alpha matte's edges and the consistency of its opaque and transparent regions. We evaluate our model across several image matting benchmarks, and the results indicate that DiffusionMat consistently outperforms existing methods.

<img src="pics/overview.png" width="800px"/>

## Set up

### Installation

```
git clone https://github.com/cnnlstm/DiffusionMat.git
cd DiffusionMat
```

### Environment

The environment can be set up  from the provided `diffusionmat.yaml`:

```
conda env create -f diffusionmat.yaml
```

## Quick Start

### Pretrained Models

Please download our pre-trained models and put in  `./pretrained_models`.

| Model | Description
| :--- | :----------
|[P3M](https://drive.google.com/file/d/1is6LEv3DjipCGjYawlbPwDPZJvTYyPdb/view?usp=sharing)  | Trained on P3M.
|[Composition-1k](https://drive.google.com/file/d/1NAuTEUGWEk3RaXWQiJTJ0m6KwpuIE1_L/view?usp=sharing)  | Trained on Composition-1k.
|[Diffusion Model](https://drive.google.com/file/d/19maZQOX5hbBM8-Jd2yVGjfcvoxh8w7dB/view?usp=sharing)  | Unconditional Alpha Matte Diffusion.
|[SwinTransformer](https://drive.google.com/file/d/1n3PhgzdMtCPJJA4mBhRjrjB4_jJbHBrd/view?usp=sharing)  | Pre-trained SwinTransformer.

### Inference

We provide 4 samples from Composition-1k dataset for the quick inference:

```
python inference.py --exp samples/alphas_pred  --config matte.yml --delta_config deltablock.yml --sample -i images --t 250 --sample_step 5 --ni
```


The whole testset of Composition-1k dataset can be downloaded at: [Composition-1k-Testset](https://drive.google.com/file/d/1fS-uh2Fi0APygd0NPjqfT7jCwUu_a_Xu/view?usp=sharing)

P3M dataset can be downloaded at: [P3M Dataset](https://drive.google.com/uc?export=download&id=1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1)

Rememer to modifying the testset path at [here](https://github.com/cnnlstm/DiffusionMat/blob/main/runners/diffusionmat_test.py#L175)

### Evaluation

Evaluate Composition-1k's results by the official evaluation MATLAB code **./DIM_evaluation_code/evaluate.m** (provided by [Deep Image Matting](https://sites.google.com/view/deepimagematting))

Evaluate P3M results by the official evaluation [Python code](https://github.com/JizhiziLi/P3M/blob/master/core/evaluate.py)



## Training

### Preparation

For obtain the trainset of Composition-1k dataset, please refer to: [Matteformer](https://github.com/webtoon/matteformer)

Please modify the trainingset path at [here](https://github.com/cnnlstm/diffusionmat/blob/df4974ba66b3f2f9c9788ce38bb87e6b2b583d33/runners/diffusionmat.py#L168)

### Start Training

```
python train.py --exp training_dir --config matte.yml --delta_config deltablock.yml --sample -i images --t 250 --sample_step 5 --ni
```

## Citation

If you find this work useful for your research, please cite:

```
@article{xu2023diffusionmat,
title={DiffusionMat: Alpha Matting as Sequential Refinement Learning},
author={Xu, Yangyang and He, Shengfeng and Shao, Wenqi and Wong, Kwan-Yee K and Qiao, Yu and Luo, Ping},
journal={arXiv preprint arXiv:2311.13535},
year={2023}
}
```

## Acknowledgement

Our Codes are mainly originated from  [SDEdit](https://github.com/ermongroup/SDEdit).
