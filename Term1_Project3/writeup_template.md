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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 100x320x3 RGB scale image                     | 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 95x315x50  |
| LeakyRELU             | 0.01 Alpha parameter                          |
| Max pooling 2x2       | 2x2 stride, valid padding, outputs 15x15x64   |
| Dropout               | Keep probability 70%                          |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 45x155x100 |
| LeakyRELU             | 0.01 Alpha paramter                           |
| Max pooling 2x2       | 2x2 stride, valid padding, outputs 20x75x100  |
| Dropout               | Keep probability 50%                          |
| Fully connected       | Layer 100                                     |
| Linear                |                                               |
| Dropout               | Keep probability 70%                          |
| Fully connected       | Layer 1                                       |
| Linear                |                                               |


My model consists of a two convolution layers followed by two fully connected dense layers. (model.py lines 34-51) 

The model includes LeakyRELU activation layers to introduce nonlinearity (code line 41,45), and the data is normalized 
in the model using a Keras lambda layer (code line 39). Carefull investigation of all images revealed that top portion
of images are less relevant for steering, thus top 60 pixel of images were not used for training (model.py code line 36).  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 43,48 and 50). Dropout layer between
convolution layers and dense layers were chosen higher values since most of the features propogated were expected to produce
noise.  

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 58-59). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

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

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
Two convolution layers with dropout layers and two dense layers. 


####3. Creation of the Training Set & Training Process

Initial attempt to train on data that recorded several laps revealed that car was not able to steer on the road where 
asphalt edges were not visible. Especially, it failed to pass curve right after the bridge, it slided into dirt and
continued as if it was road. To overcome this issue, I created another dataset where I mainly focused on steering 
on this curve.  

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to generalize steering. 
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by clipping top 60 pixels of image
and normalizing it.  

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under
fitting. The ideal number of epochs was 150 as evidenced by the training and testing losses.  
I used an stochastic gradient optimizer so that training was faster training.
