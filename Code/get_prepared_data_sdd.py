# coding=utf-8
"""Given the final dataset or the anchor dataset, compile prepared data."""

import json
import os
import operator
import pickle
import numpy as np
from tqdm import tqdm

##input: annotation ,changelst list ,dataset(videos)
##output: list output

def filter_neg_boxes(bboxes):
  new_bboxes = []
  for bbox in bboxes:
    x, y, w, h = bbox["bbox"]
    coords = x, y, x + w, y + h
    bad = False
    for o in coords:
      if o < 0:
        bad = True
    if not bad:
      new_bboxes.append(bbox)
  return new_bboxes

# be consistent with next paper, merge some classes
class2classid = {
    "person": 0,
    "car": 1,
    "bus": 1,
    "truck":1,
    "cart": 1,
    "motorcycle":1,
    "bicycle":8,
    "biker": 8,
    "skater": 8,
}  

def prepared_data_sdd(anno_data,changelst):

  target_resolution = (1920.0, 1080.0)

  def convert_bbox(bbox):
    vid, (w, h), rotated_90_clockwise = changelst

    x1, y1, x2, y2 = bbox
    if rotated_90_clockwise:
      x1, y1, x2, y2 = y1, x1, y2, x2
      x1 = w - x1
      x2 = w - x2
    # rescaling
    x1 = target_resolution[0]/w * x1
    x2 = target_resolution[0]/w * x2
    y1 = target_resolution[1]/h * y1
    y2 = target_resolution[1]/h * y2

    return [x1, y1, x2, y2]

  def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2)/2.0, (y1 + y2)/2.0

  # 1. first pass, get the needed frames
  frame_idxs = {}
  for one in anno_data:
    # is a person and not outside of view
    # one > track_id, x1, y1, x2, y2, frame_idx, lost, _, _, classname
    if (one[-1] == "person" and one[-4] == 0): #or "Person"  | lost == 0 is exist 
      frame_idxs[int(one[5])] = 1
  frame_idxs = sorted(frame_idxs.keys())

  traj_data = []  # [frame_idx, person_idx, x, y]
  frame_data = {}
  person_box_data = {}  # (frame_idx, person_id) -> boxes
  other_box_data = {}  # (frame_idx, person_id) -> other boxes + boxclasids
  for one in anno_data:
     track_id, x1, y1, x2, y2, frame_idx, lost, _, _, classname = one
     if (frame_idx not in frame_idxs) or (lost == 1):
       continue
     if frame_idx not in frame_data:
       frame_data[frame_idx] = []

     frame_data[frame_idx].append({
         "class_name": classname,
         "track_id": track_id,
         "bbox": convert_bbox([x1, y1, x2, y2])
     })
   
  for frame_idx in frame_idxs:
      box_list = frame_data[frame_idx]
      box_list.sort(key=operator.itemgetter("track_id"))

      for i, box in enumerate(box_list):
        class_name = box["class_name"]
        track_id = box["track_id"]
        bbox = box["bbox"]
        if class_name == "person":
          person_key = "%s_%d_%d" % (changelst[0], frame_idx, track_id) # changelst[0] is vid
          x, y = get_center(bbox)
          # ignore points outside of current resolution
          if (x > target_resolution[0]) or (y > target_resolution[1]):
            continue
          traj_data.append((frame_idx, float(track_id), x, y))
          person_box_data[person_key] = bbox
          
          all_other_boxes = [box_list[j]["bbox"]
                              for j in range(len(box_list)) if j != i]

          all_other_boxclassids = [class2classid[box_list[j]["class_name"]]
                                    for j in range(len(box_list)) if j != i]

          other_box_data[person_key] = (all_other_boxes,
                                        all_other_boxclassids)





  return traj_data, person_box_data, other_box_data 