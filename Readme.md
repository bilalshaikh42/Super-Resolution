# Super Resolution

## Previous work 
We start of with model developed by Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017.

The original repository is available at [https://github.com/sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)

We have clone the repository into the EDSR-PyTorch folder. Commits in this repository may include modifications to the original code.


## Datasets

The model is trained on the following datasets. More details can be found in the source repository.

The following additional dataset is used for the project. Download the dataset to the `dataset_sources` folder in a file called `brain_tumor_mri_dataset.zip`.

https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset

Then, use the following scripts to extract the datasets:

```
python brain_images.py
```

## Experiments

### Brain MRI image super-resolution
We finetune the EDSR model on the brain MRI image dataset.
