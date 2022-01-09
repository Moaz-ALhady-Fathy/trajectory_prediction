# coding=utf-8
# visualize converted annotation of SDD

import cv2

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


def visualizeSddAnnotation(dataset_resize,changelst, traj_anno, person_box_data, other_box_data,id):
  framesData={}
  classid2class = {v:k for k, v in class2classid.items()}
  for i, line in enumerate(traj_anno):
    frame_idx, track_id, x, y = line
    if track_id == id:
      output_id = f"{track_id}_{frame_idx}"
      person_key = "%s_%d_%d" % (changelst[0], frame_idx, track_id) #changelst[0] is video_id
      person_box = person_box_data[person_key]
      other_boxes, other_boxclassids = other_box_data[person_key]
      frame_data = dataset_resize[frame_idx]
      # the current person bounding box
      frame_data = cv2.rectangle(frame_data,(int(person_box[0]), int(person_box[1])),(int(person_box[2]), int(person_box[3])),
                                    color=(0, 0, 255),
                                    thickness=2)
      # the other boxes
      for box, box_classid in zip(other_boxes, other_boxclassids):
          box_label = classid2class[box_classid]
          frame_data = cv2.putText(frame_data, box_label,(int(box[0]), int(box[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5,color=(255, 0, 0))
          frame_data = cv2.rectangle(frame_data, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color=(255, 0, 0), thickness=2)
      framesData[output_id]=frame_data
  return framesData
  