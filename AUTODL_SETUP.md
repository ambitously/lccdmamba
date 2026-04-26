# AutoDL RTX 5090 Setup

This repository has been adjusted for single-GPU training and GVLM-CD256 style data:

```text
GVLM-CD256/
  A/
  B/
  label/
  list/
    train.txt
    val.txt
    test.txt
```

## 1. VMamba Files

This fork now vendors the VMamba pieces required by LCCDMamba:

```text
lccdmamba/vmamba/vmamba.py
lccdmamba/configs/config.py
lccdmamba/configs/vssm/*.yaml
third_party/selective_scan/
```

The default backbone config is `lccdmamba/configs/vssm/vmambav0_base_224.yaml`.
If you have an ImageNet-pretrained VMamba checkpoint, set it with:

```bash
export VSSM_PRETRAINED=/root/autodl-tmp/weights/vssm_base.pth
```

If this variable is not set, the backbone starts from random initialization. If you load a full LCCDMamba checkpoint for testing, the checkpoint can still overwrite the model weights.

## 2. Create Environment On AutoDL

Choose an RTX 5090 instance with an image that has a recent NVIDIA driver and CUDA 12.8 or newer. Blackwell GPUs such as RTX 5090 require PyTorch wheels built with CUDA 12.8+.

```bash
conda create -n lccdmamba python=3.11 -y
conda activate lccdmamba
python -m pip install -U pip
pip uninstall -y torch torchvision torchaudio
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd third_party/selective_scan
pip install .
cd ../..
```

Smoke test:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print((x@x).mean())"
```

If this reports an `sm_120` compatibility error, the wrong PyTorch wheel is installed. Remove old wheels and reinstall the `cu128` build.

Check the VMamba CUDA extension:

```bash
python -c "import selective_scan_cuda_oflex; print('selective_scan ok')"
```

If the extension build fails on RTX 5090, you can still run with the slower PyTorch fallback, but training will be much slower.

## 3. Upload Data

Recommended server path:

```text
/root/autodl-tmp/datasets/GVLM-CD256
```

On Windows, your local dataset path is:

```text
D:\USE\heu\paper\code\GVLM-main\GVLM-main\dataset\GVLM-CD256
```

Upload that folder to AutoDL with the same internal structure.

## 4. Run Training

Example for one RTX 5090:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --data-root /root/autodl-tmp/datasets/GVLM-CD256 \
  --dataset-name GVLM-CD256 \
  --batch-size 8 \
  --epochs 100 \
  --lr 1.4e-4 \
  --weight-decay 5e-4 \
  --num-workers 8 \
  --device 0 \
  --output-dir /root/autodl-tmp/output \
  --results-dir /root/autodl-tmp/results
```

If memory is not enough, reduce `--batch-size` to `4` or `2`. For first verification, use `--epochs 1 --batch-size 2`.

Outputs:

```text
/root/autodl-tmp/output/gvlm-cd256/<model_time>/
/root/autodl-tmp/results/GVLM-CD256/<model_time>/
```
