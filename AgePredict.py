from datetime import datetime
import sys
import os
from unittest import result
sys.path.append("./nets")
import cv2
import numpy as np
import mxnet as mx
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
from preproccessing.dataset_proc import gen_face, gen_boundbox
MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)


def load_branch(model, with_gender, use_SE, use_white_norm):
    if with_gender:
        return load_C3AE2(model, use_SE, use_white_norm)
    else:
        return load_C3AE(model, use_SE, use_white_norm)     

def load_C3AE(model, use_SE, use_white_norm):
    from nets.C3AE import build_net
    models = build_net(12, using_SE=use_SE, using_white_norm=use_white_norm)
    models.load_weights(model)
    return models

def load_C3AE2(model,  use_SE, use_white_norm):
    from nets.C3AE_expand import build_net3, model_refresh_without_nan 
    models = build_net3(12, using_SE=use_SE, using_white_norm=use_white_norm)
    if model:
        models.load_weights(model)
        model_refresh_without_nan(models) ## hot fix which occur non-scientice gpu or cpu
    return models


def predict(img_path, with_gender = False, use_SE = True, use_white_norm= True, model_path = './model/c3ae_model_v2_117_5.830443-0.955'):
    img = cv2.imread(img_path)
    models = load_branch(model_path, with_gender, use_SE, use_white_norm)


    try:
        bounds, lmarks = gen_face(MTCNN_DETECT, img, only_one=False)
        ret = MTCNN_DETECT.extract_image_chips(img, lmarks, padding=0.4)
    except Exception as ee:
        ret = None
        print(img.shape, ee)
    if not ret:
        print("no face")
        return img, None
    padding = 200
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    bounds, lmarks = bounds, lmarks

    for pidx, (box, landmarks) in enumerate(zip(bounds, lmarks)):
        trible_box = gen_boundbox(box, landmarks)
        tri_imgs = []
        for bbox in trible_box:
            bbox = bbox + padding
            h_min, w_min = bbox[0]
            h_max, w_max = bbox[1]
            #cv2.imwrite("test.jpg", new_bd_img[w_min:w_max, h_min:h_max, :])
            tri_imgs.append([cv2.resize(new_bd_img[w_min:w_max, h_min:h_max, :], (64, 64))])

        result = models.predict(tri_imgs)
        age, gender = None, None
        if result and len(result) == 3:
            age, _, gender = result
            age_label, gender_label = age[-1][-1], "F" if gender[-1][0] > gender[-1][1] else "M"
        elif result and len(result) == 2:
            age, _  = result
            age_label, gender_label = age[-1][-1], "unknown"
        else:
           raise Exception("fatal result: %s"%result)
    return (age_label, gender_label)