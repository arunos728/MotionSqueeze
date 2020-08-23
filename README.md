
# MotionSqueeze: Neural Motion Feature Learning for Video Understanding
***
<img src="/img/MS_module.png" width="50%" height="50%" alt="MS_module"></img>
***
This is the implementation of the paper "MotionSqueeze: Neural Motion Feature Learning for Video Understanding" by H.Kwon, M.Kim, S.Kwak, and M.Cho.
For more information, checkout the project [website](http://cvlab.postech.ac.kr/research/MotionSqueeze/) and the paper on [arXiv](https://arxiv.org/abs/2007.09933).

### Environment:
* Cuda: 9.0
* gcc: 5.4.0
* Python 3.6.8
* PyTorch 1.0.1
* TorchVison: 0.2.1
* Spatial Correlation Sampler (https://github.com/ClementPinard/Pytorch-Correlation-extension.git)
* Others: (https://github.com/arunos728/MotionSqueeze/conda_list.txt)

### Clone this repo

```
git clone https://github.com/arunos728/MotionSqueeze.git
```

### Running

* For training TSM or MSNet on Something-v1, use the following command:
```bash
    ./scripts/train_TSM_Something_v1.sh local
```

* For training TSM or MSNet on Kinetics, use the following command:
```bash
    ./scripts/train_TSM_Kinetics.sh local
```

* For testing your trained model on Something-v1, use the following command:
```bash
    ./scripts/test_TSM_Something_v1.sh local
```

* For testing your trained model on Kinetics, use the following command:
```bash
    ./scripts/test_TSM_Kinetics.sh local
```

### Citation
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
[Heeseung Kwon](https://github.com/arunos728/MotionSqueeze), [Manjin Kim](https://github.com/arunos728/MotionSqueeze)

Questions can also be left as issues in the repository. We will be happy to answer them.
