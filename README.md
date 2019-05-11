# Flipkart_Grid

This code is used to build a model based on ResNet-18 (no. of layers of ResNet can be changed based on system computational capacity) for **Object Localization in images**. Given an input image the code gives a bounding box around the object in image and coordinates of the bounding box. 

**Usage:**
1. In ./dataset/images directory, input images are stored.
2. In ./dataset/prediction directory, predicted output(images with bounding boxes) are stored.
3. Coordinate of bounding boxes are stored in results.csv.
4. training_set.csv has the training set images with bounding box coordinates. Bounding box coordinates are: x1 is min x coordinate , y1 is min y coordinate, x2 is max x coordinate and y2 is max y coordinate.  
To Run-   
1. For training: run getdata.py to generate id_to_box & id_to_size. Then run ./train.py
2. Pre-trained weights can be downloaded from: 
3. For Testing: To test on trained model: python test1.py  

**Dependencies:**

-python 3.6  
-tensorflow 1.3.0  
-numpy  
-PIL  
-pickle  
-matplotlib

**Neural Network Details:**

ResNet-18 is used with an additional 4 dimensional layer after it.  
Loss: smooth l1 loss  
Metric: IoU of groound truth and prediction, threshold=0.75

**Data Augmentation:**
1. Resize all images to (224, 224, 3). Here, orginal image is (480, 640,3).
2. Normalize and standardize all pixel channel.
3. The training images were Horizontally fliped, resulting an image set twice as larger as the original one.








