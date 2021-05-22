#!/bin/sh
set -eu
ROOT_DIR=${PWD}
IN_IMAGE_DIR="${ROOT_DIR}/datasets/sample_n5"
RESULTS_DIR="${ROOT_DIR}/results/sample_n5"
mkdir -p ${RESULTS_DIR}

# 学習済みモデルのダウンロード
cd ${ROOT_DIR}/api/densepose_wrapper
sh fetch_densepose_data.sh

cd ${ROOT_DIR}/api/graphonomy_wrapper
sh download_model.sh

# API 起動
cd ${ROOT_DIR}
docker-compose -f docker-compose.yml stop
docker-compose -f docker-compose.yml up -d

# 手画像を取得
sleep 10
python request.py \
    --in_image_dir ${IN_IMAGE_DIR} \
    --results_dir ${RESULTS_DIR} \
    --debug
