#!/bin/sh
set -eu
ROOT_DIR=${PWD}
DENSEPOSE_DIR=${ROOT_DIR}/DensePose
CONTAINER_NAME=densepose_container

INPUT_DIR=infer_data/sample_n5
OUTPUT_DIR=results_api/sample_n5

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/iuv
mkdir -p ${OUTPUT_DIR}/segument
mkdir -p ${OUTPUT_DIR}/contour

# データセットの準備
sh fetch_densepose_data.sh

# コンテナ起動
cd ${ROOT_DIR}
docker-compose stop
docker-compose up -d
sleep 5

# densepose の実行
cd api
python request.py \
    --port 5003 \
    --in_image_dir "../${INPUT_DIR}" \
    --results_dir "../${OUTPUT_DIR}/iuv" \
    --debug

cd ${ROOT_DIR}
sudo rm -rf ${OUTPUT_DIR}/*.pdf

# セグメント画像を取得
python visualization.py \
    --in_image_dir ${OUTPUT_DIR}/iuv \
    --results_dir "${OUTPUT_DIR}/segument" \
    --format segument

# 等高線を取得
python visualization.py \
    --in_image_dir ${OUTPUT_DIR}/iuv \
    --results_dir "${OUTPUT_DIR}/contour" \
    --format contour
