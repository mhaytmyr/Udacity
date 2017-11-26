
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
[image6]: ./examples/sliding_window.jpg "Sliding Window"
[image7]: ./examples/poly_fit_example.jpg "Fit Visual"
[image8]: ./examples/color_fit_lines.jpg "Curvature formula"
[image9]: ./examples/example_output.jpg "Output Image"
[video1]: ./project_video.mp4 "Video"


---
### Pipeline (single images)

Before working with real video I prototype pipeline on single images. Overview of pipeline is as follows:

* Undistort frame
* Apply gradient and color thresholding to find lane edges
* Convert image to top-down view (perspective transformation)
* Extract left and right lanes using histogramming method
* Fit measured points to polynomial and,
* Extract road curvature and position of vehicle

#### 1. Camera Callibration and Distortion Correction.

The code for this step is contained in the first and second code cell of the IPython notebook located in
"./examples/Pipeline\_Prototyping.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of
it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the
(x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the
`cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` 
function.

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

#### 4. Lane Identification
To find coorect lane lines I use sliding window technique. Initially, I divide image into 9 eqally spaced vertical segments.
For each segment I sum pixels in vertical axis. Then, I search for maximum in the left- and right-half of resulting image.
This gives me approximate position where lane is located. To avoid noise in each segment, I narrow search window 
to +/-80 pixels of the previously found peak center. In some cases when there is a gap between white lanes search window
fails to find peak and return first index of search window which results in the shift of window. To overcome this issue
I check for window content, if it has zero sum then I use previous known two point to propogate center (implementation
in the IPython notebook code cell 6 in the find\_window\_centroids() function)

Following picture shows one result of sliding window approach:
![alt text][image6]

In the next step I use central points from the previous step to perform polynomial fit. I use second order polynomial
to describe curvature of the lane. The code which performs this operation is on IPython notebook code cell 7. The result of 
one fit looks as follows:

![alt text][image7]

#### 5. Curvature of the raod and position of vehicle.

After fitting lane center I extract coefficients as follows. 

```latex
f(y) = A*y^2+B*y+C
```

then curvature, R is defined as 
```latex
 R = 1/|2A| * (1+(2A+B)^2)^3/2
```

The position of the vehicle is determined by the deviation center of lines with center of image. 
I use 3.7/700 m/pixels to convert this to meters. The implementation of these formulas are in the IPython notebook
code cell #7

#### 6. Final Image Output

In the final step after I have computed polynomial lines I unwrap image back to original position and color the 
detected lanes. I implemented this step in code cell #9 in IPython notebook in the function `project_fitted_line_back()`.  
Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
