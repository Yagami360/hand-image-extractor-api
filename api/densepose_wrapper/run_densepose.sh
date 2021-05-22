#!/bin/sh
set -eu
ROOT_DIR=${PWD}
DENSEPOSE_DIR=${ROOT_DIR}/DensePose
CONTAINER_NAME=densepose_container

FILE_EXT=jpg
IMAGE_FILE=infer_data/sample_n5
OUTPUT_DIR=results/sample_n5

sudo rm -rf ${OUTPUT_DIR}
sudo mkdir -p ${OUTPUT_DIR}
sudo mkdir -p ${OUTPUT_DIR}/iuv
sudo mkdir -p ${OUTPUT_DIR}/segument
sudo mkdir -p ${OUTPUT_DIR}/contour

# データセットの準備
sh fetch_densepose_data.sh

# コンテナ起動
cd ${ROOT_DIR}
docker-compose stop
docker-compose up -d

# densepose の実行
docker exec -it ${CONTAINER_NAME} /bin/bash -c \
    "python2 tools/infer_simple.py \
        --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
        --output-dir ${OUTPUT_DIR}/iuv \
        --image-ext ${FILE_EXT} \
        --wts https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
        DensePoseData/${IMAGE_FILE}"

sudo rm -rf ${OUTPUT_DIR}/*.pdf

<<COMMENTOUT
# セグメント画像を取得
sudo python visualization.py \
    --in_image_dir ${OUTPUT_DIR}/iuv \
    --results_dir "${OUTPUT_DIR}/segument" \
    --format segument

# 等高線を取得
sudo python visualization.py \
    --in_image_dir ${OUTPUT_DIR}/iuv \
    --results_dir "${OUTPUT_DIR}/contour" \
    --format contour
COMMENTOUT
