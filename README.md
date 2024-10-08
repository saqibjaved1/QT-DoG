# QT-DOG: QUANTIZATION-AWARE TRAINING FOR DOMAIN GENERALIZATION

Official PyTorch implementation of [QT-DOG: QUANTIZATION-AWARE TRAINING FOR DOMAIN
GENERALIZATION]().

QT-DoG enhances domain generalization by utilizing quantization to promote flatter minima in the loss landscape, which reduces overfitting to source domains and improves performance on unseen data. It significantly reduces model size and computational overhead without sacrificing accuracy, making it resource-efficient and suitable for real-world applications. Additionally, QT-DoG generalizes across various datasets, architectures, and quantization algorithms, and can be seamlessly combined with other domain generalization techniques, demonstrating its robustness and adaptability.


## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```


## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /my/datasets/path --quant 1 --q_steps 100 
```

Example results on PACS with ResNet-50:

| Algorithm        |   Art | Cartoon | Painting | Sketch | Avg.  |**Size** | **Models trained** |
|------------------|-------|---------|----------|--------|-------|---------|-----------------|
| **ERM (our runs)** |  89.8 |   79.7  |   96.8   |  72.5  | 84.7  |1x|  1
| **SWAD**         |  89.3 |   83.4  |   97.3   |  82.5  | 88.1  |1x| 1 |
| **EoA**          |  90.5 |   83.4  |   98.0   |  82.5  | 88.6  |6x| 6|
| **DiWA**         |  90.6 |   83.4  | **98.2** |  83.8  | 89.0  |1x| 60|
| **QT-DoG**       |  89.1 |   82.4  |   96.9   |  82.3  | 87.8  | 0.22x | 1 |
| **EoQ**          | **90.7** | **83.7** | **98.2** | **84.8** | **89.3** | 1x | 5|


In this example, QT-DoG achieves a Domain Generalization (DG) performance of 87.8% on the PACS dataset. However, when ensembling using the same method as Ensemble of Averages (EoA), our EOQ approach achieves state-of-the-art results, despite being more compact in size.


##  Results:

<p align="center">
    <img src="./assets/fig2.png" width="80%" />
</p>

###  Quantizing Vision Transformers:


| **Algorithm**           | **Backbone**    | **PACS**            | **TerraInc**        | **Compression** |
|-------------------------|-----------------|---------------------|---------------------|-----------------|
| **ERM_ViT**             | DeiT-Small      | 84.3 ± 0.2          | 43.2 ± 0.2          | -               |
| **ERM-SD_ViT**          | DeiT-Small      | **86.3 ± 0.2**       | 44.3 ± 0.2          | -               |
| **ERM_ViT + QT-DoG**    | DeiT-Small      | 86.2 ± 0.3          | **45.6 ± 0.4**       | **4.6x**        |

**Table 1:** *Quantization of a Vision Transformer:* Comparison of performance on PACS and TerraInc datasets with and without QT-DoG quantization of [ERM_ViT](https://openaccess.thecvf.com/content/ACCV2022/papers/Sultana_Self-Distilled_Vision_Transformer_for_Domain_Generalization_ACCV_2022_paper.pdf) using the DeiT-Small backbone.

###  Combination with other methods:




This project includes some code from [SWAD]([https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414](https://github.com/khanrc/swad?tab=readme-ov-file)) and [LSQ](https://github.com/zhutmost/lsq-net), also MIT licensed.
