# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Preprocess annotations for training and testing."""
import json
import numpy as np
from tqdm import tqdm

def To_npz(obs_len,pred_len,traj_data,seg_output):
  scene_h, scene_w = (36, 64)
  grid_strides="2,4"
  video_h,video_w= (1080,1920)
  scene_grid_strides = [int(o) for o in grid_strides.split(",")]
  assert scene_grid_strides
  num_scene_grid = len(scene_grid_strides)
  scene_grids = []
  for stride in scene_grid_strides:
    h, w = scene_h, scene_w
    this_h, this_w = round(h/stride), round(w/stride)
    this_h, this_w = int(this_h), int(this_w)
    scene_grids.append((this_h, this_w))

  # Get the center point for each scale's each grid
  scene_grid_centers = []
  for h, w in scene_grids:
    h_gap, w_gap = video_h/h, video_w/w
    centers_x = np.cumsum([w_gap for _ in range(w)]) - w_gap/2.0
    centers_y = np.cumsum([h_gap for _ in range(h)]) - h_gap/2.0
    centers_xx = np.tile(np.expand_dims(centers_x, axis=0), [h, 1])
    centers_yy = np.tile(np.expand_dims(centers_y, axis=1), [1, w])
    centers = np.stack((centers_xx, centers_yy), axis=-1)  # [H,W,2]
    scene_grid_centers.append(centers)

  seq_len = obs_len + pred_len

  seq_list = []  
  seq_list_rel = []
  num_person_in_start_frame = []
  seq_frameidx_list = []
  seq_key_list = [] 
  total_frame_used = {}
  seq_grid_class_list = []
  seq_grid_target_list = []  
  seq_grid_target_all_list = [[],[]]
  scene_list = []  
  scene_feat_dict = {}
  scene_key2feati = {}

  # read the json file 
  scene_id2name="scene36_64_id2name_top10.json"
  with open(scene_id2name, "r") as f:
    scene_id2name = json.load(f)  # {"oldid2new":,"id2name":}
  scene_oldid2new = scene_id2name["oldid2new"]
  scene_oldid2new = {
      int(oldi): scene_oldid2new[oldi] for oldi in scene_oldid2new}
  total_scene_class = len(scene_oldid2new)
  scene_id2name = scene_id2name["id2name"]
  assert len(scene_oldid2new) == len(scene_id2name)

  
  # videoname = "stream"
  delim = "\t"
  # get the data 
  data = []
  for line in traj_data:
    fidx, pid, x, y = line
    data.append([fidx, pid, x, y])
  data = np.array(data, dtype="float32")

  frames = np.unique(data[:, 0]).tolist()  # all frame_idx
  frame_data = []  # [num_frame, K,4]
  for frame in frames:
    frame_data.append(data[frame == data[:, 0], :])

  for idx, frame in enumerate(frames):
    cur_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
    persons_in_cur_seq = np.unique(cur_seq_data[:, 1])
    num_person_in_cur_seq = len(persons_in_cur_seq)
    # defining variables
    cur_seq = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
    cur_seq_rel = np.zeros((num_person_in_cur_seq, seq_len, 2),dtype="float32")
    cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len), dtype="int32")
    cur_seq_ids = []
    cur_seq_grids_class = np.zeros(
        (num_person_in_cur_seq, num_scene_grid, seq_len),
        dtype="int32")
    cur_seq_grids_target = np.zeros(
        (num_person_in_cur_seq, num_scene_grid, seq_len, 2),
        dtype="float32")
    cur_seq_grids_target_all = []
    for i, (h, w) in enumerate(scene_grids):
      target = np.zeros(
          (num_person_in_cur_seq, seq_len, h, w, 2),
          dtype="float32")
      cur_seq_grids_target_all.append(target)
    count_person = 0
    scene_featidx = np.zeros((num_person_in_cur_seq, seq_len, 1),
                              dtype="int")
    # this frame to the rest frame for all the persons should be the same
    frame_idxs = frames[idx:idx+seq_len]
    # get npy content
    for frame_id in frame_idxs:
      if frame_id in seg_output:
        if frame_id not in scene_key2feati:
          feati = len(scene_feat_dict.keys())
          scene_feat_dict[frame_id] = seg_output[frame_id]
          scene_key2feati[frame_id] = feati
        else:
          feati = scene_key2feati[frame_id]
      scene_featidx[:, i, :] = feati

    for person_id in persons_in_cur_seq:
      # traverse all person starting from idx frames for 20 frames

      cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

      if len(cur_person_seq) != seq_len:
        # skipping the sequence not fully cover in this frames
        """
        very
        very
        very
        very
        very important 
        """
        continue
      # [seq_len,2]
      cur_person_seq = cur_person_seq[:, 2:]
      cur_person_seq_rel = np.zeros_like(cur_person_seq)

      # first frame is zeros x,y
      cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - \
          cur_person_seq[:-1, :]

      cur_seq[count_person, :, :] = cur_person_seq
      cur_seq_rel[count_person, :, :] = cur_person_seq_rel

      # get the grid classification

      this_cur_person_seq = cur_person_seq

      # get the grid classification label based on (x,y)
      # grid centers: [H,W,2]
      for i, (center, (h, w)) in enumerate(zip(
          scene_grid_centers, scene_grids)):

        # grid classification
        h_gap, w_gap = video_h/h, video_w/w
        x_indexes = np.ceil(this_cur_person_seq[:, 0] / w_gap)  # [seq_len]

        y_indexes = np.ceil(this_cur_person_seq[:, 1] / h_gap)  # [seq_len]
        x_indexes = np.asarray(x_indexes, dtype="int")
        y_indexes = np.asarray(y_indexes, dtype="int")

        # ceil(0.0) = 0.0, we need
        x_indexes[x_indexes == 0] = 1
        y_indexes[y_indexes == 0] = 1
        x_indexes = x_indexes - 1
        y_indexes = y_indexes - 1

        one_hot = np.zeros((seq_len, h, w), dtype="uint8")
        one_hot[range(seq_len), y_indexes, x_indexes] = 1
        one_hot_flat = one_hot.reshape((seq_len, -1))  # [seq_len,h*w]
        classes = np.argmax(one_hot_flat, axis=1)  # [seq_len]
        cur_seq_grids_class[count_person, i, :] = classes

        # grid regression
        cur_person_seq_tile = np.tile(np.expand_dims(np.expand_dims(
            this_cur_person_seq, axis=1), axis=1), [1, h, w, 1])# [seq_len,h,w,2]
        center_tile = np.tile(np.expand_dims(
            center, axis=0), [seq_len, 1, 1, 1])# [seq_len,h,w,2]
        # grid_center + target -> actual xy
        # deffence between each point in seq and all points of the grid 
        all_target = cur_person_seq_tile - center_tile  # [seq_len,h,w,2]
        # only save the one grid
        cur_seq_grids_target[count_person, i, :, :] = \
            all_target[one_hot.astype("bool"), :]

        
        cur_seq_grids_target_all[i][count_person, :, :, :, :] = all_target
      
      # record the frame
      cur_seq_frame[count_person, :] = frame_idxs
      # add the seq_key
      cur_seq_ids.append("stream_%d_%d" % ( frame_idxs[0], person_id))
      count_person += 1
    # save the data
    if count_person <= 0:
      continue

    
    seq_list.append(cur_seq[:count_person])
    seq_list_rel.append(cur_seq_rel[:count_person])
    num_person_in_start_frame.append(count_person) 
    seq_frameidx_list.append(cur_seq_frame[:count_person])
    seq_key_list.append(cur_seq_ids[:count_person])

    for one in cur_seq_frame[:count_person]:
      for frameidx in one:
        total_frame_used[(frameidx)] = 1
    scene_list.append(scene_featidx[:count_person])
    seq_grid_class_list.append(cur_seq_grids_class[:count_person])
    seq_grid_target_list.append(cur_seq_grids_target[:count_person])
    for i, _ in enumerate(scene_grids):
      seq_grid_target_all_list[i].append(
          cur_seq_grids_target_all[i][:count_person])

  num_seq = len(seq_list)
  seq_list = np.concatenate(seq_list, axis=0)
  seq_list_rel = np.concatenate(seq_list_rel, axis=0)
  seq_frameidx_list = np.concatenate(seq_frameidx_list, axis=0)
  seq_key_list = np.concatenate(seq_key_list, axis=0)
  

  print("total frames %s, seq_list shape:%s, total unique frame used:%s" %
        (num_seq, seq_list.shape, len(total_frame_used)))

  obs_traj = seq_list[:, :obs_len, :]
  pred_traj = seq_list[:, obs_len:, :]

  obs_traj_rel = seq_list_rel[:, :obs_len, :]
  pred_traj_rel = seq_list_rel[:, obs_len:, :]

  # only save the obs_frames
  obs_frameidx = seq_frameidx_list[:, :obs_len]

  # the starting idx for each frame in the N*K list,
  # [num_frame, 2]
  cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
  seq_start_end = np.array([
      (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
  ], dtype="int")

  # save the data
  data = {
      "obs_traj": obs_traj,
      "obs_traj_rel": obs_traj_rel,
      "pred_traj": pred_traj,
      "pred_traj_rel": pred_traj_rel,
      "seq_start_end": seq_start_end,
      "obs_frameidx": obs_frameidx,
      "seq_key": seq_key_list #videoname_frameid_personid  first frame id
  }

  
  seq_grid_class_list = np.concatenate(seq_grid_class_list, axis=0)
  seq_grid_target_list = np.concatenate(seq_grid_target_list, axis=0)

  obs_seq_grid_class = seq_grid_class_list[:, :, :obs_len]
  obs_seq_grid_target = seq_grid_target_list[:, :, :obs_len]
  pred_seq_grid_class = seq_grid_class_list[:, :, obs_len:]
  pred_seq_grid_target = seq_grid_target_list[:, :, obs_len:]

  data.update({
      "video_wh": (video_w, video_h),
      "scene_grid_strides": scene_grid_strides,
      "obs_grid_class": obs_seq_grid_class,
      "obs_grid_target": obs_seq_grid_target,
      "pred_grid_class": pred_seq_grid_class,
      "pred_grid_target": pred_seq_grid_target,
  })
  for i, center in enumerate(scene_grid_centers):
    data.update({("grid_center_%d" % i): center})
    seq_grid_target_all_list[i] = np.concatenate(seq_grid_target_all_list[i], axis=0)
    obs_seq_grid_target_all = seq_grid_target_all_list[i][:, :obs_len, :, :, :]
    pred_seq_grid_target_all = seq_grid_target_all_list[i][:, obs_len:, :, :, :]
    data.update({("obs_grid_target_all_%d" % i): obs_seq_grid_target_all,
    ("pred_grid_target_all_%d" % i): pred_seq_grid_target_all})


  # the ids to the feature
  # [N*K, seq_len, 1]
  scene_list = np.concatenate(scene_list, axis=0)
  obs_scene = scene_list[:, :obs_len, :]
  pred_scene = scene_list[:, obs_len:, :]

  # stack all the feature into one big matrix
  # all frames in all videos # now it is jus the unique feature frame
  total_frames = len(scene_feat_dict)
  scene_feat_final_shape = (total_frames, scene_h,
                            scene_w, total_scene_class)
  print("initilizing big scene feature matrix : %s.." % list(
      scene_feat_final_shape))
  # each class will be a mask
  scene_feat_final = np.zeros(scene_feat_final_shape, dtype="uint8")
  for key in tqdm(scene_feat_dict, ascii=True):
    feati = scene_key2feati[key]
    scene_feat = scene_feat_dict[key]  # [H,W]
    # transform classid first
    new_scene_feat = np.zeros_like(scene_feat)  # zero for background class
    for i in range(scene_h):
      for j in range(scene_w):
        # rest is ignored and all put into background
        #if scene_oldid2new.has_key(scene_feat[i, j]):
        if scene_feat[i, j] in scene_oldid2new:
          new_scene_feat[i, j] = scene_oldid2new[scene_feat[i, j]]
    # transform to masks
    this_scene_feat = np.zeros(
        (scene_h, scene_w, total_scene_class), dtype="uint8")
    # so we use the H,W to index the mask feat
    # generate the index first
    h_indexes = np.repeat(np.arange(scene_h), scene_w).reshape(
        (scene_h, scene_w))
    w_indexes = np.tile(np.arange(scene_w), scene_h).reshape(
        (scene_h, scene_w))
    this_scene_feat[h_indexes, w_indexes, new_scene_feat] = 1

    scene_feat_final[feati, :, :, :] = this_scene_feat
    del this_scene_feat
    del new_scene_feat

  data.update({
      "obs_scene": obs_scene,
      "pred_scene": pred_scene,
      "scene_feat": scene_feat_final,
  })

  return data
