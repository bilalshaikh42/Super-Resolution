
base_dir=/home/ctebright/EDSR-PyTorch/

dataset=DIV2K

data_dir=$base_dir/experiment/test/results-$dataset

rm -r $base_dir/experiment/test/results-$dataset/

touch $base_dir/experiment/test/results-$dataset/PSNR_ablated_res.csv

for file in $base_dir/zeroed/*
do

    layer=$(echo $file | sed 's:.*/::')
    layer=$(echo $layer | sed 's/.pt//')

    rm $data_dir/*.png
    python main.py --data_test $dataset --pre_train $file --test_only --save_results --model EDSR --n_resblocks 32 --n_feats 256 --scale 4 --res_scale 0.1 > output.txt

    grep PSNR output.txt > temp.txt

    PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)

    echo "$layer,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_ablated_res.csv

    mkdir $data_dir/$layer
    mv $data_dir/*.png $data_dir/$layer/

    rm output.txt
    rm temp.txt

done



