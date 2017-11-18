#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/steering_distribution.png "Steering Distribution"
[image2]: ./images/image_random_brightness.png "Image Random Brightness"
[image3]: ./images/image_random_transition.png "Image Random Transition"
[image4]: ./images/image_random_flip.png "Image Random Flip"
[image5]: ./images/first_layer_features.png "Image Features"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My final model consisted of the following layers:

| Layer                 |     Description                               |    Number of Parameters  |
|:---------------------:|:---------------------------------------------:|:------------------------:|
| Input                 | 100x320x3 RGB scale image                     |          0               |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 96x316x24  |          1824            |
| LeakyRELU             | 0.01 Alpha parameter                          |           0              |
| Max pooling 2x2       | 2x2 stride, valid padding, outputs 48x158x64  |           0              |
| Dropout               | Keep probability 70%                          |           0              |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 44x155x36  |         21636            |
| LeakyRELU             | 0.01 Alpha paramter                           |           0              |
| Max pooling 2x2       | 2x2 stride, valid padding, outputs 22x77x100  |          0               |
| Dropout               | Keep probability 50%                          |           0              |
| Fully connected       | Layer 100                                     |         6098500          | 
| Linear                |                                               |           0              |
| Dropout               | Keep probability 70%                          |           0              |
| Fully connected       | Layer 1                                       |          101             |
| Linear                |                                               |


My model consists of a two convolution layers followed by two fully connected dense layers. (model.py lines 34-51)
The total number of trainable paramters are 6,122,061.  

The model includes LeakyRELU activation layers to introduce nonlinearity (code line 41,45), and the data is normalized 
in the model using a Keras lambda layer (code line 39). Carefull investigation of all images revealed that top portion
of images are less relevant for steering, thus top 60 pixel of images were not used for training (model.py code line 36).  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 43,48 and 50). Dropout layer between
convolution layers and dense layers were chosen higher values since most of the features propogated are not important.  

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 58-59). 
 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

To reduce weights stochastic gradient descent algorithm was used. This algorithm converged much faster compared to Adam 
optimizer. Learning algorithms were tuned manually by investigating loss per batches. Momentum and decay rate were 
also tuned based on the loss (model.py line 56)  

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, 
recovering from the left and right sides of the road and focusing on the road where algorithm failed the most. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with simple algorithm and add more complex layers. 

My first step was to use a convolution neural network model similar to the LeNet architecture, 
I thought this model might be appropriate because it would learn road lanes. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation
 set. I found that my first model had a low mean squared error on the training set but a high mean squared error on 
 the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that some of the layers are not used during fitting.
Then I added several dropout layers. 

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track, especially on the second curve right after the bridge. 
To improve the driving behavior in these cases, I added data which mainly focused on driving in this particular curve. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers 
and layer sizes: Two convolution layers, droput layer and one final hidden layer.  


####3. Creation of the Training Set & Training Process

Initial attempt to train on data that recorded several laps revealed that car was not able to maneuver on the road where 
asphalt edges were not visible. Especially, it failed to pass curve right after the bridge, it slided into dirt and
continued as if it was road. I also realized that distribution of steering angle was not even, it was highly 
skewed toward the central driving. Therefore, I decided to augment dataset, especailly I concentrated on the steereing 
angles that have higher values. Following image shows steering distribution before and after balancing.  

![alt text][image1]  

To balance dataset I augmented steering angles that fall to the tail of distribution as follows:
I randomly changed the brighness of camera, randomly shift image on horizontal axis or flip image with respect to verical 
axis. This would also help model better generalize steering angle. 
Following images show several of those images after each transformation:

![alt text][image2]
![alt text][image3]
![alt text][image4]

After these transformation I saved images as HDF file format. This allowed me store huge amount of data and access its 
elements efficiently. After the collection process, I had 27500 number of images. I then preprocessed this data by 
clipping top 60 pixels of image and normalizing it. I finally randomly shuffled the data set and put 
35% of the data into a validation set.  I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. Initially, I started training with 5 epochs and 
gradually increased total epoch size to 50. After each iteration I examined validation and training errors to tune 
hyperparametes or to make decition whether to continue to the next iteration. 
For the optimization algorithm I used adam optimizer with learning rate 0.001 and decay rate of 1e-5. 

After all, I vizualized if learned features make sense. I examined first layer and found out that it mostly learned edges 
of the lane, which is what I expected. Below is the example of one image from 1st convolution layer. 

![alt text][image5]



 

