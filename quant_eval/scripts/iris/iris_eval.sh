#!/bin/bash
while getopts "f": OPTION

do

  python crop_images.py --inpath ${OPTARG}/test_results/ --RA_out ${OPTARG}/real_A --RB_out ${OPTARG}/real_B --FB_out ${OPTARG}/fake_B --experiment ${OPTARG}

  python evaluation_psnr_ssim.py --real_dir ${OPTARG}/real_B --fake_dir ${OPTARG}/fake_B --experiment ${OPTARG}

done
