# favtGAN - Facial Visible Translation GAN

Here you will find information on how to download the four datasets used in the experiments and how to process them, in addition to running any of the four favtGAN implementations, and analyzing the test results using SSIM and PSNR. Models were built in PyTorch 1.6.0.

## Abstract
Thermal images reveal medically important physiological information about human stress, signs of inflammation, and emotional mood that cannot be seen on visible images. Providing a method to generate thermal faces from visible images would be highly valuable for the telemedicine community in order to show this medical information. To the best of our knowledge, there are limited works on visible-to-thermal (VT) face translation, and many current works go the opposite direction to generate visible faces from thermal surveillance images (TV) for law enforcement applications. As a result, we introduce favtGAN, a VT GAN which uses the pix2pix image translation model with an auxiliary sensor label prediction network for generating thermal faces from visible images. Since most TV methods are trained on only one data source drawn from one thermal sensor, we com- bine datasets from faces and cityscapes. These combined data are captured from similar sensors in order to boot- strap the training and transfer learning task, especially valuable because visible-thermal face datasets are limited. Experiments on these combined datasets show that favtGAN demonstrates an increase in SSIM and PSNR scores of generated thermal faces, compared to training on a single face dataset alone.

<img src='pics/fig1.png'>

<img src='pics/fig2.png'>

## Install

`git clone git@github.com:nudro/favtgan.git`

`pip install requirements.txt`

## Datasets

| Data | Num_Sensors | Train | Train Subjects | Test | Test Subjects | Total Subjects | Total Images | Eur Test IDs | Iris Test IDs |
|-|-|-|-|-|-|-|-|-|-|
| Eurecom | 1 | 945 | 45 | 105 | 5 | 50 | 1050 | 1, 2, 21, 31, 36 | n/a |
| Iris | 1 | 846 | 26 | 98 | 3 | 29 | 944 | n/a | ['Vijay', 'Meng', 'Vicky'] |
| Adas | 1 | 842 | n/a | 98 | n/a | n/a | 940 | n/a | n/a |
| OSU | 1 | 843 | n/a | 211 | n/a | n/a | 1054 | n/a | n/a |
| EA | 2 | 1787 | 45 | 203 | 5 | 50 | 1990 | 1, 2, 21, 31, 36 | n/a |
| EI | 2 | 1791 | 71 | 203 | 8 | 79 | 1994 | 1, 2, 21, 31, 36 | ['Vijay', 'Meng', 'Vicky'] |
| IO | 2 | 1689 | 26 | 309 | 3 | 29 | 1998 | n/a | ['Vijay', 'Meng', 'Vicky'] |

### Iris and OSU
The Iris and the Oklahoma State University (OSU) datasets are publicly available and free to download here:
- Iris: http://vcipl-okstate.org/pbvs/bench/Data/02/download.html
- OSU: http://vcipl-okstate.org/pbvs/bench/Data/01/download.html. We use the "1a" and "1b" sets.

Because they are publicly available, we have provided the paired visible-thermal datasets here which include Iris only, Iris + OSU, and OSU only.
- Link here: https://umbc.box.com/s/m3x7gm67wtw4nhk9qlxv1j3jq93lb0us (download only)
- There is only a train and test set, no validation.

### Eurecom and FLIR ADAS
The Eurecom and FLIR ADAS datasets must be downloaded with permission.
- Eurecom: The Eurecom dataset is accessible by permission, by filling out this form which is maintained by the researchers, Mallat et. al
http://vis-th.eurecom.fr/contact from the paper <i>Mallat, Khawla, and Jean-Luc Dugelay. "A benchmark database of visible and thermal paired face images across multiple variations." 2018 International Conference of the Biometrics Special Interest Group (BIOSIG). IEEE, 2018.</i>
- FLIR ADAS: The FLIR ADAS dataset is maintained by Flir, Inc. and can be accessed by filling out this form: https://www.flir.com/oem/adas/adas-dataset-form/

### Eurecom and Iris
To create the Eurecom and Iris dataset used in the experiments, first refer to the table above that indicate the test IDs for Eurecom and Iris. One unique individual can only exist in either the train or test set, not both. Once you have formatted Eurecom per the instructions below, you may manually move Eurecom IDs into the respective train and test sets, and combine with Iris dataset. For example, where A is the visible directory of train and test images, and B is the thermal directory.  

```
EI
- A
--train
--test
-B
--train
--test

```

You may then use the official pix2pix pairing script to combine them: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/combine_A_and_B.py. Detailed instructions are provided here: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md, and also in the Jupyter Notebooks provided in the Dataset Preparation steps below to help guide you.

## Dataset Preparation
Eurecom and ADAS require preprocessing which can be found in `/dataset_proc/`.

Train/Test Splits: Not all the FLIR ADAS data is used for experiments, since there is a 10:1 ratio of ADAS:Eurecom. We randomly selected files from ADAS in order to achieve a closer 1:1 balance with the Eurecom dataset. As a result, see `/dataset_proc/train_EA_files.txt` and `/dataset_proc/test_EA_files.txt` to ensure the correct ADAS files have been split into the train and test files.

For Eurecom, you will need to manually ensure that the test set contains test IDs {1, 2, 21, 31, 36} and the training set contains the other 45. Note there is only a train and test set, no validation.

After downloading the FLIR ADAS dataset, use `/dataset_proc/FLIR_ADAS_Preproc.ipynb` to guide you through formatting ADAS. Again, ensure that the correct FLIR ADAS files are used in the combined Eurecom + ADAS dataset by referring to the `/dataset_proc/train_EA_files.txt` and `/dataset_proc/test_EA_files.txt`.

After downloading the Eurecom dataset, use `dataset_proc/EURECOM_Prep.ipynb` to convert `.tiff` to `.jpg` thermal images. The visible images from Eurecom come as `.jpg` files already. The notebook will contain instructions on how to label the files and place them into a visible and thermal directory, respectively. At this point, I would suggest splitting them into their respective train and test sets, based on the test IDs shown in above and in the table.


## Models

### favtGAN
We provide the four favtGAN implementations:

- Baseline under `favtGAN/favtGAN/pix2pix-smooth-baseline.py` (smooth indicates label smoothing has been applied in the script to convert 1.0 to 0.99)
- No Noise, `favtGAN/favtGAN/pix2pix-smooth-no-noise.py`
- Noisy Labels, `favtGAN/favtGAN/pix2pix-noisy-label.py``
- Gaussian, `favtGAN/favtGAN/pix2pix-gaussian.py`

The dataloader class, `favtGAN/favtGAN/datasets.py` and a test script `favtGAN/favtGAN/test.py` is also provided.

You may run `bash train_EI_sensor_baseline.sh` to train favtGAN baseline on the EI dataset:

```
python pix2pix-smooth-baseline.py --dataset_name EI
 --annots_csv labels/EI_s.csv
 --n_epochs 2000
 --batch_size 12
 --gpu_num 0
 --out_file EI_sensor_baseline
 --sample_interval 100000
 --checkpoint_interval 10
 --experiment EI_sensor_baseline

 ```

Labels are provided under `labels` that are the class labels for each thermal sensor provided for `EA_s.csv`, `EI_s.csv`, and `IO_s.csv`

To train ADAS + Eurecom (EA), you may modify the bash scripts for any of the four implementations, or you can simply run:

```
python pix2pix-smooth-baseline.py --dataset_name EA
 --annots_csv labels/EA_s.csv
 --n_epochs 2000
 --batch_size 12
 --gpu_num 0
 --out_file EA_sensor_baseline
 --sample_interval 100000
 --checkpoint_interval 10
 --experiment EA_sensor_baseline
```
A log file will be written to .txt file during training. Further, .pth saved checkpoints will be saved under `saved_models` and samples from the test set will be stored in `images` during training.


Manually create a new directory under `images` called `images/test_results` to store the generated thermal faces. Then run: `bash test.sh`.

### pix2pix

We provide the pix2pix scripts we used for comparison from the https://github.com/eriklindernoren/PyTorch-GAN#pix2pix repository also provided which are minimal models based on the official pix2pix repository. They are located in:

`favtGAN/pix2pix/pix2pix-smooth.py` and the test script at `favtGAN/pix2pix/test.py`.

To train for EI, Eurecom-only, and Iris-only, respectively, you can run the provided bash scripts:

`train_EI.sh`

`train_eur.sh`

`train_iris.sh`


## Evaluation
We use SSIM and PSNR to measure image quality and similarity against the test results.

### Crop:
First, the test results generated need to be split into separate real visible (A), real thermal (B), and fake thermal (B) images for evaluation. For the Eurecom test results, first run:

```
python quant_eval/scripts/eurecom/crop_images.py
  --in_path your/test/results/dir
  --RA_out real_A
  --RB_out real_B
  --FB_out fake_B
```
Note that the script will make directories automatically in where `opt.experiment` is the name of the experiment previously run such as `EI_sensor_baseline` under training.

```
os.makedirs("quant_eval/Eurecom/%s/fake_B" % opt.experiment, exist_ok=True)
os.makedirs("quant_eval/Eurecom/%s/real_B" % opt.experiment, exist_ok=True)
os.makedirs("quant_eval/Eurecom/%s/real_A" % opt.experiment, exist_ok=True)
```

### PSNR and SSIM:
Next, after the crops have been made, run the below. It will output a psnr.csv and ssim.csv file, and place it under the experiment directory.

```
bash quant_eval/scripts/eurecom/eurecom_eval.sh -f "EI_sensor_baseline"
```

The result should be a directory structure like this:

```
Eurecom/EI_sensor_baseline
- fake_B
- real_A
- real_B
- test_results
- psnr.csv
- ssim.csv
```

### Analyze
Two notebooks are provided to calculate the mean of each experiment's PSNR and SSIM scores for Eurecom and Iris.

<hr />
For questions contact cordun1@umbc.edu.
