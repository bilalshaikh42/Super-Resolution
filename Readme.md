# Super Resolution

## Previous work 
We start of with model developed by Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017.

The original repository is available at [https://github.com/sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)

We have clone the repository into the EDSR-PyTorch folder. Commits in this repository may include modifications to the original code.


## Datasets

The model is trained on the following datasets. More details can be found in the source repository.

[DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf):download from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).

All benchmark datasets can be downloaded [here](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB). 

Extract the needed datasets with the following commands: 
```bash
tar -xvf dataset_sources/benchmark.tar ./dataset
tar -xvf dataset_sources/DIV2K.tar ./dataset
```
The following additional dataset is used for the project. Download the dataset to the `dataset_sources` folder in a file called `brain_tumor_mri_dataset.zip`.

[Brain Tumor MRI dataset](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset)

Then, use the following scripts to extract the datasets:

```
python brain_images.py
```


## Pre Training 
To train the original models, use the following command:

x2 model
```
 python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2  
```
x4 model (requires x2 model to be trained)

```
  python main.py --model EDSR --scale 4 --save edsr_baseline_x4  --pre_train ../experiment/edsr_baseline_x2/model/model_best.pt
```

## Fine tuning
To fine tune the original models for the brain tumor dataset, use the following command:
```
python main.py --model EDSR --scale 4 --save edsr_brain_x4   --pre_train ../models/edsr_baseline_x4-6b446fab.pt --data_train BrainTumor --data_test BrainTumor --save_results 
```

The results from our training can be loaded by extracting the appropriate archive from the `experiment` folder.
## Experiments

Test the fine tuned model on the brain tumor dataset.
```
   python main.py --model EDSR --scale 4 --save edsr_brain_x4_eval   --pre_train ../models/brain_model_best.pt --data_train BrainTumor --data_test BrainTumor --save_results --test_only --self_ensemble
```
Test the fine tuned model on the DIV2K dataset.
```
python main.py --model EDSR --scale 4 --save edsr_brain_x4_eval   --pre_train ../models/brain_model_best.pt  --save_results --test_only --self_ensemble 
```

Test the original model on the Brain Tumor dataset.
```
python main.py --model EDSR --scale 4 --save edsr_baseline_x4_eval   --pre_train ../models/edsr_baseline_x4-6b446fab.pt  --save_results --test_only --self_ensemble --data_test BrainTumor
```

Test the original model on the DIV2K dataset.
```
python main.py --model EDSR --scale 4 --save edsr_baseline_x4_eval   --pre_train ../models/edsr_baseline_x4-6b446fab.pt  --save_results --test_only --self_ensemble
```
