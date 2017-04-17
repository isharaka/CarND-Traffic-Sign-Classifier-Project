#**Traffic Sign Recognition** 


[//]: # (Image References)

[image1]: ./images/data_visualization.jpg "Visualization"
[image2]: ./images/grayscaling.jpg "Grayscaling"
[image3]: ./images/softmax_visualization.jpg "Softmax Visualization"
[image4]: ./web/web1.jpeg "Traffic Sign 1"
[image5]: ./web/web2.jpeg "Traffic Sign 2"
[image6]: ./web/web3.jpeg "Traffic Sign 3"
[image7]: ./web/web4.jpeg "Traffic Sign 4"
[image8]: ./web/web5.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/isharaka/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
Here is the [html version](https://github.com/isharaka/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is  32 x 32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.
The bar charts show the distribution of images over the classes in training, validation and testing datasets.

- The relative distrbution in all three sets is similar. This indcates the spliting of data in to sets has been done randomly as required.
- Some classes (traffic signs) have more examples than others. This may effect the model so that it will learn to detect some classes better than others.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because, during testing I consistently got slightly better results with grayscale images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I skipped normalizing since it produced inferior results. This was unexpected and could possibly have been due to numerical errors.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the datasets provided. The details of the different sets are described above.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| |Layer         		|     Description	        					| 
|:|---:---------------------:|:---------------------------------------------:| 
|1| Input         		| 32x32x1 Grayscale image   							| 
|| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
|| RELU					|												|
|| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x16 				|
|2| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32|
|| RELU					|												|
|| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x32 				|
|| Dropout					|  dropout probability 0.5	|
3| Fully connected		| input 800, output 400									|
|| RELU					|												|
|4| Fully connected		| input 400, output 200									|
|| RELU					|												|
|5| Fully connected		| input 200, output 120									|
|| RELU					|												|
|6| Fully connected		| input 120, output 43									|
|| Softmax				|         									|
|| Cross entropy				|         									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) . This makes use of the momentum of parameters during the serach for optimum parameters, where as a simple gradient descent optimizer does not. Therefore Adam optimizer will converge faster.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
training set accuracy of 99.9%
validation set accuracy of 94.9%
test set accuracy of 94.6%

*What architecture was chosen?*
[LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)  

*Why did you believe it would be relevant to the traffic sign application?*
During the lab project LeNet achived lose to 96% accuracy in classifying somewhat similar (similar in size of the input images and general shape of the content) MNIST dataset.

*How was the architecture adjusted and why was it adjusted? *
The adjustments are summarized below.

| Adjustment			        |     Rationale	        					| 
|:---------------------:|:---------------------------------------------:| 
| Deepen the input convolutional layers|MNIST data consists of b&w images where as traffic signs have colour images. <br> Therefore traffic sign images have more features for the input layers to learn|
| Add an extra fully connected layer|Traffic sign classifier must learn to classify 4 times the classes (43 to 10). <br> Therefore it was necessary to increase the size of the network. <br>|
|Add a droput element|Adding a droput element before the first fully connected layer reduced <br> overfitting. There is published [evidence](https://arxiv.org/pdf/1506.02158v6.pdf)  that adding  dropout elements <br> to all layers in LeNet yields better results. However addition of droput elements <br>to multple layers caused severe underfitting indicating the need to <br>enlarge the network.  This was not practical due to resource constraints |

*What were some problems with the initial architecture?*
LeNet uses max pooling to reduce the faeture map sizes. This removes 75% of the information in the feature map. 

*What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?*

- Convolutional layers help classification by sharing weights over the spatial dimensions of the image. This is expected help the model recognize the traffic signs regardless of the location of teh sign within the image.
- Dropout regularizer help the classification by preventing overfitting. It forces the network to learn redundant paths for each class. During testing, adding droput elements narrowed the distance beween training accuracy and validation accuracy.

*Which parameters were tuned? How were they adjusted and why?*

| Adjustment			        |     Rationale	        					| 
|:---------------------:|:---------------------------------------------:| 
| batch size 64| batch sizes lower and higher produced poorer results during testing|
| learning rate lowered to 0.0001| in order to improve the fit of the model. <br>with a higher rate the accuracy oscillated with each epoch <br> after reaching a certain level|
| epochs increased to 50| in order to allow sufficient time to converge |

*How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?*

All three accuracies are above 90% and they are close to each other. Training accuracy is approximately 5% higher indicating some degree of overfitting.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)     		| Speed limit (60km/h)									| 
| Priority road     			| Priority road										|
|  Right-of-way at the next intersection					|  Right-of-way at the next intersection											|
| Speed limit (50km/h)	      		| Speed limit (50km/h)				 				|
| Stop			| Stop   							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 
This compares favorably to the accuracy on the test set of 94.6%. 

However during different runs the accuracy of the web traffic sign prediction were sometimes 80%. Most of the times classifying of the speed signs failed by classifying a speed limit sign as a different speed sign. i.e. 50km/h was sometimes classfied as 60km/h or 80km/h.

Approximately 15% difference in accuracy between the test set and the images from the web, however, is not a good representation of the model's perfromance. This is due to the small sample size of 5.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for visualizing the certainity of predictions on my final model is located in the 11th cell of the Ipython notebook.

The softmax probabilities for the top 5 predictions for each image is listed below. The closeness of the softmax probabilities in the top 5 prediction for speed signs highlights the modles poor performance in classifying the speed signs. The model shows higher degree of certainity relative to other potential prediction when classifying signs that are not speed limits. 
This could be due to the insuffcicent resolution of the image for the model to identify numerals correctly.

Expected prediction:  3  Speed limit (60km/h)

| Prediction	  |     Softmax Prob. | Sign		| 
|:------:|:---------------:|:---------------------------------------------:| 
|  3   | 0.30   | Speed limit (60km/h)|
|  2   | 0.22   | Speed limit (50km/h)|
|  5   | 0.18   | Speed limit (80km/h)|
|  1   | 0.17   | Speed limit (30km/h)|
|  4   | 0.13   | Speed limit (70km/h)|
 
Expected prediction: 12  Priority road

| Prediction	  |     Softmax Prob. | Sign		| 
|:------:|:---------------:|:---------------------------------------------:| 
| 12   | 0.34   | Priority road|
| 40   | 0.25   | Roundabout mandatory|
| 38   | 0.16   | Keep right|
| 15   | 0.14   | No vehicles|
|  8   | 0.12   | Speed limit (120km/h)|
 
Expected prediction: 11  Right-of-way at the next intersection

| Prediction	  |     Softmax Prob. | Sign		| 
|:------:|:---------------:|:---------------------------------------------:| 
| 11   | 0.33  |  Right-of-way at the next intersection|
| 28   | 0.22 |   Children crossing|
| 30   | 0.18  |  Beware of ice/snow|
| 23   | 0.14   | Slippery road|
| 27   | 0.13   | Pedestrians|
 
Expected prediction:  2  Speed limit (50km/h)

| Prediction	  |     Softmax Prob. | Sign		| 
|:------:|:---------------:|:---------------------------------------------:| 
|  2   | 0.28   | Speed limit (50km/h)|
|  3   | 0.19   | Speed limit (60km/h)|
|  5   | 0.18   | Speed limit (80km/h)|
|  1   | 0.18   | Speed limit (30km/h)|
|  7   | 0.17   | Speed limit (100km/h)|
 
Expected prediction: 14  Stop

| Prediction	  |     Softmax Prob. | Sign		| 
|:------:|:---------------:|:---------------------------------------------:| 
| 14   | 0.44   | Stop|
| 38   | 0.22   | Keep right|
| 13   | 0.14   | Yield|
| 40   | 0.12   | Roundabout mandatory|
| 34   | 0.07   | Turn left ahead|

The same data is graphically visualized here.
![alt text][image3]