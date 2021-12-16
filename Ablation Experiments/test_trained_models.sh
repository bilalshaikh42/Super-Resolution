
base_dir=/home/ctebright/EDSR-PyTorch/

dataset=DIV2K
data_dir=$base_dir/experiment/test/results-$dataset

rm -r $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

touch $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv


mkdir /home/ctebright/EDSR-PyTorch/my_trained_models/

cp /home/ctebright/EDSR-PyTorch/experiment/edsr_baseline_x4_16blocks/model/model_best.pt /home/ctebright/EDSR-PyTorch/my_trained_models/16blocks_best.pt
cp /home/ctebright/EDSR-PyTorch/experiment/edsr_baseline_x4_16blocks/model/model_latest.pt /home/ctebright/EDSR-PyTorch/my_trained_models/16blocks_latest.pt

cp /home/ctebright/EDSR-PyTorch/experiment/edsr_baseline_x4_8blocks/model/model_best.pt /home/ctebright/EDSR-PyTorch/my_trained_models/8blocks_best.pt
cp /home/ctebright/EDSR-PyTorch/experiment/edsr_baseline_x4_8blocks/model/model_latest.pt /home/ctebright/EDSR-PyTorch/my_trained_models/8blocks_latest.pt

cp /home/ctebright/EDSR-PyTorch/experiment/edsr_baseline_x4_4blocks/model/model_best.pt /home/ctebright/EDSR-PyTorch/my_trained_models/4blocks_best.pt
cp /home/ctebright/EDSR-PyTorch/experiment/edsr_baseline_x4_4blocks/model/model_latest.pt /home/ctebright/EDSR-PyTorch/my_trained_models/4blocks_latest.pt

#all trained models
name=4blocks_best
file=$name.pt

python main.py --data_test $dataset --pre_train /home/ctebright/EDSR-PyTorch/my_trained_models/$file --test_only --save_results --model EDSR --n_resblocks 4 --scale 4 --res_scale 0.1 > output.txt

grep PSNR output.txt > temp.txt

PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)

echo "$name,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

mkdir $data_dir/$name
mv $data_dir/*.png $data_dir/$name/

rm output.txt
rm temp.txt



name=4blocks_latest
file=$name.pt

python main.py --data_test $dataset --pre_train /home/ctebright/EDSR-PyTorch/my_trained_models/$file --test_only --save_results --model EDSR --n_resblocks 4 --scale 4 --res_scale 0.1 > output.txt

grep PSNR output.txt > temp.txt

PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)


echo "$name,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

mkdir $data_dir/$name
mv $data_dir/*.png $data_dir/$name/

rm output.txt
rm temp.txt




name=8blocks_best
file=$name.pt

python main.py --data_test $dataset --pre_train /home/ctebright/EDSR-PyTorch/my_trained_models/$file --test_only --save_results --model EDSR --n_resblocks 8 --scale 4 --res_scale 0.1 > output.txt

grep PSNR output.txt > temp.txt

PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)


echo "$name,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

mkdir $data_dir/$name
mv $data_dir/*.png $data_dir/$name/

rm output.txt
rm temp.txt



name=8blocks_latest
file=$name.pt

python main.py --data_test $dataset --pre_train /home/ctebright/EDSR-PyTorch/my_trained_models/$file --test_only --save_results --model EDSR --n_resblocks 8 --scale 4 --res_scale 0.1 > output.txt

grep PSNR output.txt > temp.txt

PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)


echo "$name,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

mkdir $data_dir/$name
mv $data_dir/*.png $data_dir/$name/

rm output.txt
rm temp.txt


name=16blocks_best
file=$name.pt

python main.py --data_test $dataset --pre_train /home/ctebright/EDSR-PyTorch/my_trained_models/$file --test_only --save_results --model EDSR --n_resblocks 16 --scale 4 --res_scale 0.1 > output.txt

grep PSNR output.txt > temp.txt

PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)

echo "$name,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

mkdir $data_dir/$name
mv $data_dir/*.png $data_dir/$name/

rm output.txt
rm temp.txt



name=16blocks_latest
file=$name.pt

python main.py --data_test $dataset --pre_train /home/ctebright/EDSR-PyTorch/my_trained_models/$file --test_only --save_results --model EDSR --n_resblocks 16 --scale 4 --res_scale 0.1 > output.txt

grep PSNR output.txt > temp.txt

PSNR=$(sed -e 's/.*PSNR: \(.*\)(Best.*/\1/' temp.txt)


echo "$name,$PSNR" >> $base_dir/experiment/test/results-$dataset/PSNR_retrained_res.csv

mkdir $data_dir/$name
mv $data_dir/*.png $data_dir/$name/

rm output.txt
rm temp.txt
