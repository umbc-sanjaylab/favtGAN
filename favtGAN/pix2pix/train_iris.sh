set -ex
python pix2pix-smooth.py --dataset_name iris_v3_pairs --n_epochs 2000 --batch_size 12 --gpu_num 1 --out_file iris_V3 --sample_interval 200 --checkpoint_interval 10 --experiment iris_V3