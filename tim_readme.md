Timothy Isonio
tisonio3@gatech.edu

# Classifier

Code for finetuning a classifier to work on MRI images is found in brain_tumor_classification.ipynb
Need to have downloaded the MRI dataset and ran the brain_images.py script to create LR and HR versions.

Copy LR versions to the ./EDSR-PyTorch/data folder
Go to ./EDSR-PyTorch/src and run the command

python main.py --data_test Demo --pre_train [YOUR MODEL HERE.pt] --test_only --save_results --model EDSR --n_resblocks [NUM BLOCKS] --scale 4 --res_scale [RES SCALE]

filling in the blanks as appropriate
that will create SR versions of the input images using the model you provide. Move those to a folder that the notebook can see

The images must be in folders according to their class, e.g.
./train/gl/gl001.png
./test/pi/pi022.png

luckily they all have their label in their filename, so you can do something like

mkdir gl
mkdir pi
mkdir me
mv *gl* gl/
mv *pi* pi/
mv *me* me/

Then edit the third cell of the notebook as appropriate and let it run and output loss/accuracy curves.
