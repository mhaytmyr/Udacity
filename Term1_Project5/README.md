
# Vehicle Detection Project

[//]: # (Image References)

[image1]: ./examples/car_not_car.jpg "Data Examples"
[image2]: ./examples/HOG_example.jpg "HOG Features"
[image3]: ./examples/learning_cruve.png "Learning Curve"
[image4]: ./examples/sliding_windows.jpg "Sliding Window Search"
[image5]: ./examples/multi_scale_windows.jpg "Multi-Scale Search"
[image7]: ./examples/bboxes_and_heat.jpg "Bounding Box and Heatmap"
[image8]: ./examples/output_bboxes.png "Final Output"
[video1]: ./output_images/project_video.mp4 "Video"


In this project, I wrote a software pipeline to detect vehicles in a video stream. 

---
### Pipeline (single images)

The goals / steps of this project are as following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### 1. Datasets and exploration.

To train a classifier I used labeled dataset where cars are cropped from the street images. For the class there are images
of roadsides that are not cars. In the dataset total number of positive labels are 8797 and negative classes is 8970, this 
is balanced dataset. Each image contains RGB color with 64x64 size. 

The code for this step is contained in the third and fourth code cell of IPython notebook located in 
"Project5\_Pipeline.ipynb". 

To demonstrate this step, I will show some of the labeled images from the dataset:
![alt text][image1]

#### 2. Feature Extraction and Training. 

To train classfier I extract image features using Histogram of Oriented Gradients technique using a 
skimage.feature.hog package. HOG counts occurences of gradients orientation in localized portions of the image. The features
are computed in dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy. 
In this analysis, I choose 9 orientations for gradients and cell sizes of 8x8 where normalization is peerformed in 2x2 cell
blocks. This resulted in 6108 features. 

The implementation is described on the code cell of 5 in the IPython notebook "Project5\_Pipeline.ipynb". 
Following image shows some of the randomly picked images and their HOG features.
![alt text][image2]

I combine cars and non-cars dataset to single dataframe and perform normalization. Then, using SVM classifier model is 
trained. To avoid overfitting "L2" regularization is used. I split data into 30% testing and 70% training datasets. Using 
70% data I peerform 10-fold cross-validation training. With this I got about 92% cross-validation score, and 93% testing score.

The plot of learning curve can be found below:
![alt text][image3]

Once I am confident about the performance of classifier, I save all hyper-parameters to pickle file. 
I also save normalization parameters which I use for pre-processing. These parameters are used to normalize images 
from video stream. 

#### 3. Sliding Window Search

The classifier was trained on 64x64 pacthes of images. However, the images from the dashboard camera is different. Therefore, 
exhaustive search is necessary to detect vehicles. I test my classifier on some of the images as follows:
![alt text][image4]

As can be seen there are a lot of false positives, especially when the selected image patch contains significant 
amount of gradients (trees). Hence, I have decided to narrow down search window to contain only lower-half of the screen 
and perform multiple scale search. Then, for each overlapping window I added them together to produce heatmap. 
The resulting image of multiple scales and combined heatmap looks as follow:
![alt text][image5]

Muti-scale search sometimes resulted in false positives more often. Hence, I decided to take intersection of different scales
to reduce false positives. This significanly reduced number of false positives however, it also narrowed down bounding box
of real images. 
![alt text][image6]
![alt text][image7]

#### 4. Final Output Image

Finally, I use heatmaps from the previous step and label them. To accomplish this task I used 
scipy.ndimage.measurements.label package. Label package uses connectivity graph to combine neighboring pixles that have 
similar intensities. The implementation this step is described in the code cell of 5 in the IPython notebook 
"Project5\_Pipeline.ipynb". Then, bounding box is drawn for each discovered label. The final output from single
processed image looks as follows
![alt text][image8]

#### 5. Summary

In this project I developed a software pipeline to track vehicles. Initially, I used labeled dataset to train SVM classifier
to identify patches of image that contains vehicles. To extract features I used histogram of oriented gradients. This 
allowed me to build robust classifier. In the next step, I wrote a function to exustively search whole image space for 
vehicles. I repeated last step with three different scales to improve accuracy of the detection. I realized after this step
algorihtm produced a lot of false positives. To avoid this I took intersection of three scales. 
This process significanly reduced false detection, however, bounding boxes shrink in size. 
Current algorithm is not efficient as it takes about 3-4 seconds per frame, which is not acceptable for online driving. 
I think there is a room for improvement in HOG features, maybe, we don't need all features. Moreover, there exist 
much more sophisticated and efficient algorithms that can segment images. 




