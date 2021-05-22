#!/bin/sh
set -eu
ROOT_DIR=${PWD}
DENSEPOSE_DIR=${ROOT_DIR}/DensePose
CONTAINER_NAME=densepose_container

FILE_EXT=jpg
IMAGE_FILE=infer_data/sample_n5
OUTPUT_DIR=results/sample_n5_keypoints

sudo mkdir -p ${OUTPUT_DIR}
sudo rm -rf ${OUTPUT_DIR}

# データセットの準備
sh fetch_densepose_data.sh

# コンテナ起動
cd ${ROOT_DIR}
docker-compose stop
docker-compose up -d

# densepose の実行
# https://github.com/facebookresearch/DensePose/blob/master/MODEL_ZOO.md
docker exec -it ${CONTAINER_NAME} /bin/bash -c \
    "python2 tools/infer_simple.py \
        --cfg configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml \
        --output-dir ${OUTPUT_DIR} \
        --image-ext ${FILE_EXT} \
        --wts https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl \
        DensePoseData/${IMAGE_FILE}"

