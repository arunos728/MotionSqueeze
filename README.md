
### Environment:
* Cuda: 9.0
* gcc: 5.4.0
* Python 3.6.8
* PyTorch 1.0.1
* TorchVison: 0.2.1
* Spatial Correlation Sampler (https://github.com/ClementPinard/Pytorch-Correlation-extension.git)

### Clone this repo

```
git clone https://github.com/arunos728/MS_online_demo.git
```

### Training
<!-- 1. Download the initialization and trained models:

```Shell
      sh models/download_models.sh
```

* If you can not access Google Drive, please download the pretrained models from [BaiduYun](https://pan.baidu.com/s/1Hx52akJLR_ISfX406bkIog), and put them in "models" folder.
-->

Command for running model:

```bash
    ./scripts/run_TSM_Something_v1.sh local
```

<!--
3. For training C3D network use the following command:

```bash
    ./scripts/run_c3dres_kinetics.sh local
```

4. For finetuning on UCF101 use the following command:

```bash
    ./scripts/run_ECOLite_finetune_UCF101.sh local
```
 -->
### NOTE
<!-- * If you want to train your model from scratch change the config as following:
```bash
    --pretrained_parts scratch
```
* configurations explained in "opts.py" -->

#### TODO
<!-- 1. ECO Full
2. Trained models on other datasets -->


#### Citation
<!-- If you use this code or ideas from the paper for your research, please cite our paper:
```
@inproceedings{ECO_eccv18,
author={Mohammadreza Zolfaghari and
               Kamaljeet Singh and
               Thomas Brox},
title={{ECO:} Efficient Convolutional Network for Online Video Understanding},	       
booktitle={ECCV},
year={2018}
}
``` -->

#### Contact
<!-- 
  [Mohammadreza Zolfaghari](https://github.com/mzolfaghari/ECO-pytorch), [Can Zhang](https://github.com/zhang-can/ECO-pytorch)

  Questions can also be left as issues in the repository. We will be happy to answer them. -->
