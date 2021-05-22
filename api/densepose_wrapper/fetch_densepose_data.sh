#!/bin/sh
set -eu
ROOT_DIR=${PWD}
DENSEPOSE_DIR=${ROOT_DIR}/DensePose

# データセットの準備
cd ${DENSEPOSE_DIR}/DensePoseData
if [ ! -e ${DENSEPOSE_DIR}/DensePoseData/DensePose_COCO ] ; then
    bash get_DensePose_COCO.sh
fi
if [ ! -e ${DENSEPOSE_DIR}/DensePoseData/UV_data ] ; then
    bash get_densepose_uv.sh
fi
if [ ! -e ${DENSEPOSE_DIR}/DensePoseData/eval_data ] ; then
    bash get_eval_data.sh
fi
