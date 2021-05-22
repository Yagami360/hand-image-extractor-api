#!/bin/sh
set -eu
IN_IMAGE_DIR=sample_n5
RESULTS_DIR=results
if [ ! -e "checkpoints/universal_trained.pth" ] ; then
    sh download_model.sh
fi
if [ -d "${RESULTS_DIR}" ] ; then
    rm -r ${RESULTS_DIR}
fi

python inference_all.py \
    --device gpu \
    --in_image_dir ${IN_IMAGE_DIR} \
    --results_dir ${RESULTS_DIR} \
    --load_checkpoints_path checkpoints/universal_trained.pth \
    --save_vis \
    --use_amp \
    --debug
