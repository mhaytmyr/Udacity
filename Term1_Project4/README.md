
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/test_undist.jpg "Undistorted"
[image2]: ./examples/road_undistort.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/hls_color_space.jpg "HLS Color Space"
[image5]: ./examples/warped_straight_lines.jpg "Warp Example"
[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first and second code cell of the IPython notebook located in 
"./examples/Pipeline\_Prototyping.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each 
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of 
it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the 
(x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the 
`cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

Before working with real video I prototype pipeline on single images. Overview of pipeline is as follows:

* Undistort frame
* Apply gradient and color thresholding to find lane edges
* Convert image to top-down view (perspective transformation)
* Extract left and right lanes using histogramming method
* Fit measured points to polynomial and,
* Extract road curvature and position of vehicle

#### 1. Distortion correction pipeline.

In the previous step, after callibration I store necessary parameters as pickle file so that I don't have to repeat
same procedure for each image. The cv2.calibrateCamera() returns camera matrix and distortion coefficients. 
Camera matrix describes intrinsic properties of camera such as focal length (f\_x, f\_{y}) and optical centers 
(c\_{x},c\_{y}). Distortion coefficients are size 5 array, of which first 2 corresponds to radial distortion and 
next two are tangental distortion coefficients. 

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Converting image to binary.

I used a combination of color and gradient thresholds to generate a binary image: (thresholding steps are 
in the 3rd code cell of "Pipeline\_Prototyping.ipynb" notebook).

* First, based on solely color information. I use HSV color spectrum to highlight yellow and white spots.
* Second, I convert resulting image to HLS spectrum and choose LS channel to compute gradients. I realized S channel is 
specifically good at identifying yellow lanes. 

I optimize thresholds based on the several test images. Here's an example of my output for this step.  

![alt text][image3]
and result of HLS color spectra:
![alt text][image4]

#### 3. Perspective transform. 

The code for my perspective transform includes a function called `corners_unwarp()`, 
which appears in the 5th code cell of the IPython notebook.  The `corners_unwarp()` function takes as inputs an image (`img`),
as well as distortion matrix (`mtx`) and distortion paramteters (`dist`). Initially, I undistor image using pre-saved 
correction matrices. Then, I use source and destinations to unwarp undistorted image. I chose the hardcode the source and 
destination points in the following manner:

```python
src = np.float32([ [595, 450], [685, 450], [1110, image.shape[0]], [210, image.shape[0]] ])
dst = np.float32([ [340, 0], [image.shape[1]-340, 0], [image.shape[1]-340, image.shape[0]], [340, image.shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 340, 0        | 
| 210, 720      | 340, 720      |
| 1110, 720     | 940, 720      |
| 685, 450      | 940, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and 
its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
