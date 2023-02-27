# Synthetic Tumors Make AI Segment Tumors Better

This repository provides extensive examples of synthetic liver tumors generated by our novel strategies. Check to see if you could tell which is real tumor and which is synthetic tumor. More importantly, our synthetic tumors can be used for training AI models, and have proven to achieve a similar (actually, *better*) performance in real tumor segmentation than a model trained on real tumors. 

**Amazing**, right? 

<p align="center"><img width="100%" src="figures/VisualTuringTest.png" /></p>

## Paper

<b>Label-Free Liver Tumor Segmentation</b> <br/>
Qixin Hu<sup>1</sup>, [Yixiong Chen](https://scholar.google.com/citations?hl=en&user=bVHYVXQAAAAJ)<sup>2</sup>, [Junfei Xiao](https://lambert-x.github.io/)<sup>3</sup>, Shuwen Sun<sup>4</sup>, [Jie-Neng Chen](https://scholar.google.com/citations?hl=en&user=yLYj88sAAAAJ)<sup>3</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>3</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>3,*</sup> <br/>
<sup>1 </sup>Huazhong University of Science and Technology,  <sup>2 </sup>Fudan University,  <br/>
<sup>3 </sup>Johns Hopkins University,   <sup>4 </sup>The First Affiliated Hospital of Nanjing Medical University <br/>
CVPR, 2023 <br/>
paper | [code](https://github.com/MrGiovanni/SyntheticTumors) | slides | demo

<b>Synthetic Tumors Make AI Segment Tumors Better</b> <br/>
Qixin Hu<sup>1</sup>, [Junfei Xiao](https://lambert-x.github.io/)<sup>2</sup>, [Yixiong Chen](https://scholar.google.com/citations?hl=en&user=bVHYVXQAAAAJ)<sup>3</sup>, Shuwen Sun<sup>4</sup>, [Jie-Neng Chen](https://scholar.google.com/citations?hl=en&user=yLYj88sAAAAJ)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>2,*</sup> <br/>
<sup>1 </sup>Huazhong University of Science and Technology,   <sup>2 </sup>Johns Hopkins University, <br/>
<sup>3 </sup>Fudan University,   <sup>4 </sup>The First Affiliated Hospital of Nanjing Medical University <br/>
Medical Imaging Meets NeurIPS, 2022 <br/>
[paper](https://arxiv.org/pdf/2210.14845.pdf) | [code](https://github.com/MrGiovanni/SyntheticTumors) | slides | demo

## 0. Preparation

```bash
git clone https://github.com/MrGiovanni/SyntheticTumors.git
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
```

#### Dataset

please download these datasets and save to `<data-path>` (user-defined).

- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)
- 05 [Label](https://www.dropbox.com/s/8e3hlza16vor05s/label.zip)

```bash
wget https://www.dropbox.com/s/jnv74utwh99ikus/01_Multi-Atlas_Labeling.tar.gz # 01 Multi-Atlas_Labeling.tar.gz (1.53 GB)
wget https://www.dropbox.com/s/5yzdzb7el9r3o9i/02_TCIA_Pancreas-CT.tar.gz # 02 TCIA_Pancreas-CT.tar.gz (7.51 GB)
wget https://www.dropbox.com/s/lzrhirei2t2vuwg/03_CHAOS.tar.gz # 03 CHAOS.tar.gz (925.3 MB)
wget https://www.dropbox.com/s/2i19kuw7qewzo6q/04_LiTS.tar.gz # 04 LiTS.tar.gz (17.42 GB)
wget https://www.dropbox.com/s/8e3hlza16vor05s/label.zip
```

#### Dependency
The code is tested on `python 3.8, Pytorch 1.11`.
```bash
conda create -n syn python=3.8
source activate syn (or conda activate syn)
pip install external/surface-distance
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## 1. Train Swin UNETR using real liver tumors

```
# UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap 0.2 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --logdir="runs/lits_real.pretrain.swin_unetrv2_base" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/lits.json --use_pretrained
# UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap 0.2 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12232 --cache_num=200 --logdir="runs/lits_real.no_pretrain.swin_unetrv2_base" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/lits.json
# UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --val_overlap 0.2 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12233 --cache_num=200 --logdir="runs/lits_real.no_pretrain.swin_unetrv2_small" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/lits.json
# UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --val_overlap 0.2 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=200 --logdir="runs/lits_real.no_pretrain.swin_unetrv2_tiny" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/lits.json
```

## 2. Train Swin UNETR using synthetic liver tumors

```
# UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --val_overlap=0.2 --syn --logdir="runs/lits_synthetic.pretrain.swin_unetrv2_base" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/healthy.json --use_pretrained
# UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --val_overlap=0.2 --syn --logdir="runs/lits_synthetic.no_pretrain.swin_unetrv2_base" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/healthy.json
# UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12233 --cache_num=200 --val_overlap=0.2 --syn --logdir="runs/lits_synthetic.no_pretrain.swin_unetrv2_small" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/healthy.json
# UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=4000 --save_checkpoint --workers=12 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=240 --val_overlap=0.2 --syn --logdir="runs/lits_synthetic.no_pretrain.swin_unetrv2_tiny" --train_dir <data-path> --val_dir <data-path> --json_dir datafolds/healthy.json
```

## 3. Evaluation

#### Swin UNETR trained by real tumors

```
# UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_real.pretrain.swin_unetrv2_base --save_dir out
# UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_real.no_pretrain.swin_unetrv2_base --save_dir out
# UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_real.no_pretrain.swin_unetrv2_small --save_dir out
# UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_real.no_pretrain.swin_unetrv2_tiny --save_dir out
```

#### Swin UNETR trained by synthetic tumors

```
# UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_synthetic.pretrain.swin_unetrv2_base --save_dir out
# UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_synthetic.no_pretrain.swin_unetrv2_base --save_dir out
# UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_synthetic.no_pretrain.swin_unetrv2_small --save_dir out
# UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir <data-path> --json_dir datafolds/lits.json --log_dir runs/lits_synthetic.no_pretrain.swin_unetrv2_tiny --save_dir out
```

## TODO

- [x] Upload the paper to arxiv
- [ ] Make a video about Visual Turing Test (will appear in YouTube)
- [ ] Make an online app for Visual Turing Test
- [x] Apply for a US patent

## Citation

```
@article{hu2022synthetic,
  title={Synthetic Tumors Make AI Segment Tumors Better},
  author={Hu, Qixin and Xiao, Junfei and Chen, Yixiong and Sun, Shuwen and Chen, Jie-Neng and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2210.14845},
  year={2022}
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research. The segmentation model is based on [Swin UNETR](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb); we appreciate the effort of the authors for providing open source code to the community.
