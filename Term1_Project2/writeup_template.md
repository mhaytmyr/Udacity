***Traffic Sign Recognition*** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_distribution_original.png "Visualization"
[image2]: ./examples/rgb_to_grayscale.png "Grayscaling"
[image3]: ./examples/augmented_images.png "Random augmentation"
[image4]: ./examples/class_distribution_uniform.png "Visualization 2"
[image5]: ./examples/General_caution.png "Traffic Sign 1"
[image6]: ./examples/Roundabout_mandatory.png "Traffic Sign 2"
[image7]: ./examples/Speed_limit_(60km_h).png "Traffic Sign 3"
[image8]: ./examples/Stop.png "Traffic Sign 4"
[image9]: ./examples/Go_straight_or_left.png "Traffic Sign 5"
[image10]: ./examples/Bicycles_crossing.png "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
***Data Set Summary & Exploration***

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data classes are distributed
on training, validation and testing datasets respectively. 

![alt text][image1]

***Design and Test a Model Architecture***

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because perfromance was better compared to color images. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because activation functions are not sensitive to large numbers.

I decided to generate additional data because training was not able to generalize to unreperesented classes.  

To add more data to the the data set, I used three image modification techniques:
* Randomly change contrast between 0.2 and 1.8, defined in tf.image.random_contrast
* Randomly change brightness between -30 and 30, tf.image.random_brighness
* Randomly rotate image between -20 degrees and 20 degrees, tf.contrib.image.rotate

Here is an example of an original image and an augmented image:

![alt text][image3]

With this procedure I was able to increase training dataset size to 107006.  
The class distribution in the training dataset become more uniform. 

Here is class distribution after augmentation.  
![alt text][image4]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x10 	|
| RELU					|												|
| Max pooling 2x2	  	| 2x2 stride, valid padding, outputs 15x15x64 	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x32	|
| RELU                  |                                               |
| Max pooling 2x2       | 2x2 stride, valid padding, outputs 6x6x32     |
| Dropout               | Keep probability 60%                          |
| Fully connected		| Layer 120        								|
| Softmax				|         									    |
| Fully connected       | Layer 84                                      |
| Softmax				|												|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an "Adam" optimizer with learning rate of 0.01 and reqularization parameter of 0.001. 
The bacth size of 8000 is used to train in 30 epoch's. I optimized hyperparamters by iteratively, training and calculating
validation accuracy. I plotted loss and accuarcy for each iteration to diagnose learning performance and overfitting.    

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97%
* validation set accuracy of 95% 
* test set accuracy of 92%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
I started with LeNet model.  
* What were some problems with the initial architecture? 
This architecture performed well in training dataset. However, it was not able to generelize on validation dataset. 
Moreover, as I continued training validation loss started increasing which suggested overfitting was issue.  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
I added one layer and one parameter to reduce overfitting. On the convolution layers I added L2 regularization term, and before the first dense layer I added droput layer. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layers help to identify correlation among the neighboring cells. Droput layer was usef because after the 
second convolution layer there are a lot of nodes which are not important, thus droput layer helps to remove them. 

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

***Test a Model on New Images***

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because they are zoomed in and occupy full image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      	| General Caution								| 
| Roundabout      		| Roundabout 									|
| Speed limit (60km/h)	| Speed limit (60 km/h)							|
| Stop	      			| Stop	                 		 				|
| Go straight or left	| Go straight or left                    		|
| Bicycles crossing     | Slippery road                                 |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 50%. 
This is not compares favorably to the accuracy on the test set. The model needs to be improved. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is general caution sign (probability of 0.99), 
and the image contains general caution sign. The top three soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General Caution								| 
| .00     				| Dangerous curve to the right					|
| .00					| Traffic signals       					    |


For the second image probability is small, however it still predicted right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .65         			| Roundabout mandatory							| 
| .16     				| Road work 									|
| .07					| Go straight or left					        |
| .04                   | Priority road                                 |
| .04                   | Go straight or right                          |

For the third image the model successfully predicted there is speed limit.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .47         			| Speed limit (60km/h)							| 
| .30     				| Speed limit (30km/h)							|
| .15					| Speed limit (20km/h)					        |
| .06                   | Road work                                     |
| .01                   | Speed limit (80km/h)                          |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


