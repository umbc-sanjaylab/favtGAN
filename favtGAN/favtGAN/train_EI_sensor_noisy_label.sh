set -ex
python pix2pix-smooth-noisy-label.py --dataset_name EI --annots_csv labels/EI_s.csv --n_epochs 2000 --batch_size 12 --gpu_num 0 --out_file EI_sensor_noisy-label --sample_interval 100000 --checkpoint_interval 10 --experiment EI_sensor_noisy-label --lambda_adv 0.8
