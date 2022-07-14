export datasets_dir=/PATH/TO/DATA
export model=resnet18
export w_bits=4
export a_bits=8

python main.py --data_path $datasets_dir --arch $model --n_bits_w $w_bits --channel_wise --n_bits_a $a_bits --act_quant --test_before_calibration
