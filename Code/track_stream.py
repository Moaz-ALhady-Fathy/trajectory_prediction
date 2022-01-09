# limit the number of cpus used by high performance libraries
import sys
sys.path.insert(0, './yolov5')


import glob
import hashlib
import json
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import numpy as np
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from yolov5.utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from yolov5.utils.torch_utils import torch_distributed_zero_first


from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn




def detect(images_try):
    annotation=[]
    dataset_resize={}
    target_resolution = (1920, 1080)
    
    
    yolo_weights= 'yolov5l.pt'
    config_deepsort= 'deep_sort_pytorch/configs/deep_sort.yaml'
    imgsz= [640]
    imgsz *= 2 if len(imgsz) == 1 else 1
    half = False
    dnn=False
    device=''

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn)
    stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
        

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    for frame_idx, image_try in enumerate(images_try):

        # Padded resize
        frame_img = letterbox(image_try, 640, 32, True)[0]

        # Convert
        frame_img = frame_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame_img = np.ascontiguousarray(frame_img)


        img, im0s, vid_cap, s = frame_img, image_try, None, ''
        # for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        #     # LOGGER.info(f"{vid_cap}")

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred,.4,.5, classes=[0, 1, 2, 3, 5, 7],agnostic=False,max_det=1000)
        dt[2] += time_sync() - t3
        
        # LOGGER.info(f"DIDN'T DETECT {pred} !!!")
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            s += '%gx%g ' % img.shape[2:]  # print string
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2]
                        y2 = output[3]

                        annotation.append([id,x1,y1,x2,y2,frame_idx,0,-1,-1,names[c]])
            else:
                deepsort.increment_ages()
                
            #Resize 
            video_id="stream"
            h,w = im0s.shape[0:2]
            rotate_90_clockwise = False
            
            if h > w:
                rotate_90_clockwise = True

            if rotate_90_clockwise:
                im0s = cv2.transpose(im0s)
            dataset_resize[frame_idx] = cv2.resize(im0s, target_resolution)

    changelst=[video_id, (w,h), rotate_90_clockwise]
    
        

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    return dataset_resize,changelst , annotation
