# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
from PIL import Image
import numpy as np

from caffe2.python import workspace

sys.path.append(os.path.join(os.getcwd(), '../DensePose'))
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)

def inference( cfg_path, weights, img_pillow, output_dir ):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(cfg_path)
    #print( "cfg : ", cfg )
    assert_and_infer_cfg(cache_urls=False, make_immutable=False)

    cfg.NUM_GPUS = 1
    weights = cache_url(weights, cfg.DOWNLOAD_CACHE)
    model = infer_engine.initialize_model_from_cfg(weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    im_name = "test"
    #img_cv = cv2.imread(im_name)
    img_np = np.asarray(img_pillow)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
            model, img_cv, None, timers=timers
        )

    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

    vis_utils.vis_one_image(
        img_cv[:, :, ::-1],  # BGR -> RGB for visualization
        im_name,
        output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        cls_bodys,
        dataset=dummy_coco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
        kp_thresh=2
    )

    IUV_SaveName = os.path.basename(im_name).split('.')[0]+'_IUV.png'
    INDS_SaveName = os.path.basename(im_name).split('.')[0]+'_INDS.png'
    iuv_pillow = Image.open(os.path.join(output_dir, '{}'.format(IUV_SaveName)))
    inds_pillow = Image.open(os.path.join(output_dir, '{}'.format(INDS_SaveName)))
    return iuv_pillow, inds_pillow
