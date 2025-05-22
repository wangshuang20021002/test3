<!-- ## Method

<p align="center"> <img src='assets/balpoe_calibrated_framework.png' align="center"> </p>

## Abstract
Many real-world recognition problems are characterized by long-tailed label distributions. These distributions make representation learning highly challenging due to limited generalization over the tail classes. If the test distribution differs from the training distribution, e.g. uniform versus long-tailed, the problem of the distribution shift needs to be addressed. A recent line of work proposes learning multiple diverse experts to tackle this issue. Ensemble diversity is encouraged by various techniques, e.g. by specializing different experts in the head and the tail classes. In this work, we take an analytical approach and extend the notion of logit adjustment to ensembles to form a Balanced Product of Experts (BalPoE). BalPoE combines a family of experts with different test-time target distributions, generalizing several previous approaches. We show how to properly define these distributions and combine the experts in order to achieve unbiased predictions, by proving that the ensemble is Fisher-consistent for minimizing the balanced error. Our theoretical analysis shows that our balanced ensemble requires calibrated experts, which we achieve in practice using mixup. We conduct extensive experiments and our method obtains new state-of-the-art results on three long-tailed datasets: CIFAR-100-LT, ImageNet-LT, and iNaturalist-2018. -->

## Getting Started

### Prerequisites

IPE is built on pytorch and a handful of other open-source libraries.

To install the required packages, you can create a conda environment:

```sh
conda create --name ipe python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Hardware requirements
4 GPUs with >= 24G GPU RAM are recommended (for large datasets). Otherwise, the model with more experts may not fit in, especially on datasets with more classes (the FC layers will be large). We do not support CPU training at the moment.

## Datasets
### Four benchmark datasets
* Please download these datasets and put them to the /data file.
* CIFAR-100 / CIFAR-10 will be downloaded automatically with the dataloader.
* iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).
* ImageNet-LT can be found at [here](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).

### Txt files
* We provide txt files for long-tailed recognition under multiple test distributions for ImageNet-LT and iNaturalist 2018. CIFAR-100 will be generated automatically with the code.
* For iNaturalist 2018, please unzip the iNaturalist_train.zip.
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_backward2.txt
│   ├── ImageNet_LT_backward5.txt
│   ├── ImageNet_LT_backward10.txt
│   ├── ImageNet_LT_backward25.txt
│   ├── ImageNet_LT_backward50.txt
│   ├── ImageNet_LT_forward2.txt
│   ├── ImageNet_LT_forward5.txt
│   ├── ImageNet_LT_forward10.txt
│   ├── ImageNet_LT_forward25.txt
│   ├── ImageNet_LT_forward50.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_uniform.txt
│   └── ImageNet_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_backward2.txt
    ├── iNaturalist18_backward3.txt
    ├── iNaturalist18_forward2.txt
    ├── iNaturalist18_forward3.txt
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```

## Usage

### CIFAR100-LT 
#### Training

* Important: to reproduce our main results, train five runs with SEED = {1,2,3,4,5} and compute mean and standard deviation over reported results.

* To train IPE with three experts on the standard-training regime, run this command:
```
python train.py -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json --seed 1
```
* One can change the imbalance ratio from 100 to 10/50 by changing the config file. Similar instructions for CIFAR10-LT.

* To get the sets of preference vectors, run this command:
```
python t_get_ray.py -r [checkpoint_path] -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json
```
* where checkpoint_path should be of the form CHECKPOINT_DIR/checkpoint-epoch[LAST_EPOCH].pth.

* To train the DNN with sets of preference vectors, run this command:
```
python train_stage2.py
```
* The trained model will be saved as dnn_text_to_vector.pth


#### Evaluate
* To evaluate IPE on diverse test class distributions, run:
``` 
python test_finally.py -r checkpoint_path [--posthoc_bias_correction]
```
where checkpoint_path should be of the form CHECKPOINT_DIR/checkpoint-epoch[LAST_EPOCH].pth.

Optional: use --posthoc_bias_correction to adjust logits with known test prior.



## Citation


## Acknowledgements

Our codebase is based on several open-source projects, particularly: 
- [SADE](https://github.com/Vanint/SADE-AgnosticLT) 
- [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition)
