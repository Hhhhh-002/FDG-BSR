# FDG-BSR

Source code accompanying **Factorized Degradation-Guided Blind Image Super-Resolution for Complex Degradation Scenarios**.

The implementation includes the Hierarchical High-Order Random Degradation Model (HHRD), the Factorized Degradation-Guided Super-Resolution Generator (FDRG), and the Texture-Structure Coordinated Discriminator (TSCD).

## Repository contents

```text
FDG-BSR/
├── bsr/
│   ├── data/                  # Image datasets and HHRD degradation
│   ├── models/                # FDG-BSR generator, TSCD, and shared modules
│   ├── contrastive.py         # Contrastive learning components
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
├── train.py                   # Training entry point
├── test.py                    # Inference entry point
├── calculate_metrics.py       # PSNR and LPIPS evaluation
└── requirements.txt
```

## Installation

Use Python 3.10 or newer. From the repository root, install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset download links

### DIV2K train HR

[Download DIV2K_train_HR.zip](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)

### Flickr2K

[Download Flickr2K.tar](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

### OST / Outdoor Scenes

- [Google Drive](https://drive.google.com/drive/folders/1LIb631GU3bOyQVTeuALesD8_eoApNniB?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1OHLfHobCcALCXut61CynXg)

The datasets are hosted by third parties and are not included in this repository. Please follow the original providers' usage conditions and cite the corresponding dataset papers.

## Prepare training images

Extract the downloaded datasets and place the HR images used for training under a common directory, for example:

```text
data/train_hr/
├── DIV2K/
├── Flickr2K/
└── OST/
```

The training loader searches the specified directory recursively. It generates LR training inputs from HR images using HHRD. To match the manuscript's experiments, use the dataset subsets and training settings specified in the manuscript.

## Training

An example invocation for stage 2 is:

```bash
python train.py --model fdg-bsr --train-hr data/train_hr --stage 2 --work-dir runs/fdg_bsr_stage2 --device cuda
```

The training script supports three stages:

1. `--stage 1`: contrastive degradation-encoder pretraining.
2. `--stage 2`: restoration training with reconstruction and contrastive losses.
3. `--stage 3`: adversarial fine-tuning with TSCD.

Use `--resume path/to/checkpoint.pth` to initialize from a saved checkpoint. Checkpoints are written to `--work-dir`. Run `python train.py --help` for all available options. The example command illustrates the interface; it does not specify a complete reproduction schedule.

## Inference

Supply a trained checkpoint and a directory of LR images:

```bash
python test.py --model fdg-bsr --checkpoint path/to/checkpoint.pth --input data/test_lr --output results/fdg_bsr --scale 4 --device cuda
```

Reconstructed images are saved with an `_SR.png` suffix. Set architecture options such as `--channels` and `--num-groups` to match the checkpoint. Pretrained checkpoints are not included in this repository; inference without `--checkpoint` uses randomly initialized weights.

## Evaluation

```bash
python calculate_metrics.py --sr-dir results/fdg_bsr --hr-dir data/test_hr --device cuda
```

The evaluation script computes PSNR on RGB images in the range `[0, 1]` and LPIPS with the AlexNet backbone. It matches SR and HR images by filename stem after removing the `_SR` and `_HR` suffixes. Each matched pair must have the same image dimensions. Match the manuscript's preprocessing and evaluation protocol when comparing results.

The examples use CUDA. The scripts also accept `--device cpu`.
