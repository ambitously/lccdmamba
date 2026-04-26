#  LCCDMamba: Visual State Space Model for Land Cover Change Detection of VHR Remote Sensing Images

 IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing

## This Fork

`main.py` now supports configurable single-GPU training:

```bash
python main.py --data-root /path/to/GVLM-CD256 --dataset-name GVLM-CD256 --batch-size 8 --epochs 100 --device 0
```

GVLM-CD256 is supported in its native layout: `A/`, `B/`, `label/`, and `list/train.txt`, `list/val.txt`, `list/test.txt`.

See `AUTODL_SETUP.md` for RTX 5090 / AutoDL environment setup.

This fork vendors the VMamba model/config files required by `lccdmamba/backbone.py`. The optional VMamba ImageNet-pretrained checkpoint can be supplied with the `VSSM_PRETRAINED` environment variable.


The model weights can be found at: [GoogleCloud](https://drive.google.com/drive/folders/1fercsG25CGvukqRFluwELrcAyCQhA2c6?usp=sharing)

```
@ARTICLE{10845192,
  author={Huang, Junqing and Yuan, Xiaochen and Lam, Chan-Tong and Wang, Yapeng and Xia, Min},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={LCCDMamba: Visual State Space Model for Land Cover Change Detection of VHR Remote Sensing Images}, 
  year={2025},
  volume={18},
  number={},
  pages={5765-5781},
  keywords={Feature extraction;Transformers;Remote sensing;Computational modeling;Land surface;Accuracy;Convolution;Correlation;Visualization;Data models;Global features;land cover change detection (LCCD);local features;mamba;VHR images},
  doi={10.1109/JSTARS.2025.3531499}}
```

