# Pedestrian Trajectory Prediction
Predicting future trajectories of pedestrians in cameras of novel scenarios and views.

## This repository contains the code and models for the following ECCV'20 paper:

[ SimAug: Learning Robust Representatios from Simulation for Trajectory Prediction ](https://arxiv.org/abs/2004.02022)
Junwei Liang, Lu Jiang, Alexander Hauptmann

## Our Pipeline

<p align="center">
  <img width="500" src="Images/pipline.jpg" >
</p>

•	**Input**: could be from a streaming camera or saved videos.

•	**Detection**: we used a pre-trained model called YOLO (You Only Look Once) to perform object detection, it uses convolutional neural networks to provide real-time object detection, it is popular for its speed and accuracy.

•	**Tracking**: we used a pre-trained model called Deep SORT (Simple Online and Realtime Tracking), it uses deep learning to perform object tracking in videos. It works by computing deep features for every bounding box and using the similarity between deep features to also factor into the tracking logic. It is known to work perfectly with YOLO and also popular for its speed and accuracy.

•	**Resizing**: at this step, we get the frames and resize them to the required shape which is 1920X 1080.

•	**Semantic Segmentation**: we used a pre-trained model called Deep Lab (Deep Labeling) an algorithm made by Google, to perform the semantic segmentation task, this model works by assigning a predicted value for each pixel in an image or video with the help of deep neural network support. It performs a pixel-wise classification where each pixel is labeled by predicted value encoding its semantic class.

•	**SimAug Model**: Simulation as Augmentation, is a novel simulation data augmentation method for trajectory prediction. It augments the representation such that it is robust to the variances in semantic scenes and camera views, it predicts the trajectory in unseen camera views.

•	**Predicted Trajectory**: The output of the proposed pipeline.


## Dependencies
•	Python 3.6 ; TensorFlow 1.15.0 ; Pytorch 1.7 ; Cuda 10

## Code Contributors
<a href="https://github.com/Moaz-ALhady-Fathy/trajectory_prediction/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Moaz-ALhady-Fathy/trajectory_prediction" />
</a>

## References
```
@inproceedings{liang2020simaug,
  title={SimAug: Learning Robust Representations from Simulation for Trajectory Prediction},
  author={Liang, Junwei and Jiang, Lu and Hauptmann, Alexander},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {August},
  year={2020}
}
```
