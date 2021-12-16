

echo "starting 4 res blocks"
python main.py --model EDSR --save edsr_baseline_x4_4blocks --model EDSR --n_resblocks 4 --scale 4

echo "starting 8 res blocks"
python main.py --model EDSR --save edsr_baseline_x4_8blocks --model EDSR --n_resblocks 8 --scale 4

echo "starting 16 res blocks"
python main.py --model EDSR --save edsr_baseline_x4_16blocks --model EDSR --n_resblocks 16 --scale 4

