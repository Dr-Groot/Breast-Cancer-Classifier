# Breast-Cancer-Classifier
The aim is to categorize and accurately identify the types and subtypes of breast cancer

Download Dataset [DOWNLOAD](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/download)

INTRODUCTION

In Breast Cancer Classification with Keras we are using mod-ern day method, technologies, processes and new ways of thinking of solving a problem in an accurate and effective manner by using techniques which are inspired by the work-ing of human brain and biological neural network and serves to improve Artificial Intelligence and which is applied in many fields of computer vision, speech recognition, natural lan-guage processing, drug design and audio recognition which is Deep Learning.

![image](https://user-images.githubusercontent.com/63160825/157168132-4bd22dbe-17e6-4209-9a9a-50d61097da25.png)

Along with ML (Machine Learning) we are using Keras which is an open source neural network library written in python. Keras is a high level API which can run on the top of Theano (optimising compiler which are excessed using NumPy-esque syn-tax),CNTK (Microsoft Cognitive Toolkit) and TenserFlow (open source software library). It is all about enabling fast prototyp-ing and experimentation while seamlessly running on Graph-ical Process Unit and Computer Processing Unit. Along with that it is an extensible, modular and user friendly.

As we discussed about technologies and libraries which we will be used, now let's have a look towards where we will be using these technology.

![image](https://user-images.githubusercontent.com/63160825/157168235-ccd62924-45f8-4952-a68b-27b44fa42755.png)

Cancer which forms in the cells of the breast is called Breast Cancer. It can occur in female but rarely occurs in male, which is visible from above mentioned World Health Organisa-tion's report. Most common breast cancer is IDC (Invasive Ductal Carcinoma). IDC is a type of cancer which develops in the milk duct, invading fat breast tissues outside the duct. Breast cancer symptoms are lump in the breast, blood from the nipple and change in the texture or shape of a breast. Its treatment depends upon cancer stage and may consist of ra-diation, chemotherapy, surgery and hormone therapy. Here we are going to make a breast cancer classifier which can accurately classify IDC (Invasive Ductal Carcinoma) data set containing histology image (histology is all about studying mi-croscopic structure of tissue) as malignant (infectious) or be-nign (not harmful) with the help of Deep Learning an

My aim is to categorise and accurately identify the types and sub types of breast can-cer which is an important clinical task and some automated methods should be used for saving time and reducing error.

Basically in this version of project we build a classifier for training a breast cancer histology data set from about 80% data set. Beside all this we have taken almost 10% of data set for the validation purpose. We have defined CNN (Convolutional Neural Network) using Keras we call it CancerNet and our images in the data sets are trained by it. The model performance is analysed by deriving confusion matrix.

This is was the brief about my project. This idea came when I read an article where back in 2012-13, NCI (National Cancer Institute) and NIH (National Institutes of Health) were develop-ing a suite of processing image and algorithms of Machine learning to automatically analyse breast histology images for risk factor in cancer, a problem or task that take professional pathologist hours for completion but there work helped facili-tates, further advancement in breast cancer factor prediction back at that time  when a machine learning was not so popu-lar or mainstream as it is now. 

![image](https://user-images.githubusercontent.com/63160825/157168588-72848dc5-693c-46f8-9cdc-a904406a47fe.png)

To analyse the cellular structure in the breast histology image they were instead of leveraging vision of basic computer and image algorithm processing but combining them these algo-rithm work really well but required a quite bit of work to put together. Today I thought of exploring breast cancer classifi-cation using DL (Deep Learning). As an aspiring data scien-tist, engineer and practitioner. It’s important for us to gain hand on to apply experience in deep learning and to comput-er vision and medical, this will help in developing a deep learn-ing algorithm for better aid to pathologist in predicting cancer. 

# Hardware and Software

## Hardware:
-	**Central Processing Unit (CPU) :** Minimum of Intel Core i5 6th Generation or AMD equivalents processor are also optimal.
-	**RAM**: About 16 GB is recommended but 8 GB is mini-mum.
-	**Graphic Processing Unit (GPU):** NVIDIA GeForce GTX960 or higher but AMD GPU’s are not comfortable with deep learning.
- **Operating System:**  Ubuntu or Microsoft Windows 10. Not use MacBook Air as it takes allot of time for training the model else MacBook Pro is better.

## Software:
-	**Anaconda:** Python interpreter flavour comes in two type  for python 2.7 and python 3.6
-	**Python:** Programming language use for scripting the Deep Learning algorithm.

# Technology Used

![image](https://user-images.githubusercontent.com/63160825/157168734-fde54534-f2c4-4f21-9b6d-d1a01fcfd173.png)

-	**MEDICAL COMPUTER VISION:** Computer Vision helps medical professional in saving their most valuable time on some fundamental or basic task but also in saving patient’s life. The technology application used for the medical to per-form by extending current ways that it is already being used and adding a layer’s of creativity and imagination. Here we are using medical computer vision in solving breast cancer classification in an appropriate and accurate manner.

-	**PYTHON:** Python is a high level, interpreted, general pur-pose programming language. It is created by Guido van Rossum and was released in 1991. It is a garbage collected and is dynamically typed as it also support multiple pro-gramming paradigm including structured, OOP (Object Ori-ented Programming language) and functional programming. Here we will be using python for scripting Deep Learning model as for splitting data set into different sets -training, validation and testing and also for training our model.

-	**KERAS:** Keras which is an open source neural network li-brary written in python. Keras is a high level API which can run on the top of Theano (optimising compiler which are ex-cessed using NumPy-esque syntax),CNTK (Microsoft Cognitive Toolkit) and TenserFlow (open source software li-brary). It is all about enabling fast prototyping and experi-mentation while seamlessly running on Graphical Process Unit and Computer Processing Unit along with that it is ex-tensible, modular and user friendly. We are using Keras se-quential API’s to build CancerNet.  

![image](https://user-images.githubusercontent.com/63160825/157168803-2e67deef-fbf9-47e0-89f4-0804ce557f61.png)


-	**TenserFlow & DEEP LEARNING:** TenserFlow is created by Google for creating deep learning models. Deep Learning is used in multi layer neural network and is a category of Ma-chine Learning. It is basically used for building complex ap-plication with high accuracy. Using Deep Learning in this project we are splitting the data set and training the data set for accurate result as it will lead  to more work with less time including complex problem.

![image](https://user-images.githubusercontent.com/63160825/157168818-175a5f67-c048-430f-8627-0125ecc20424.png)

# DATABASE

Machine learning or Deep Learning can not yield any accu-rate results until it is trained. For training we need previous re-ports or set of data (Database) for which we have to train our model according to the situation or problem that we have to solve. So by this you came to know why Database is so im-portant for using Deep Learning algorithms.

Our problem is to deal with healthcare, specially Breast Can-cer and we came up to a solution that is designing a classifier for Breast Cancer Prediction using Deep Learning with Keras so for that we need allot of data so we are using data set for IDC (Invasive Ductal Carcinoma), the most common of Breast Cancer.

![image](https://user-images.githubusercontent.com/63160825/157169289-6eb360d8-6b6d-48f0-a670-0b9359fdf903.png)

The data set was originally curate by **Janowczyk**, **Madabhu-shi** and **Roa** et al but it is publicly available at Kaggle’s web-site. The original set of data is having **162 slide image scanned at 40x**.

Slide image’s are naturally big (in terms of territorial dimen-sions), so making them easier to work we take all total **2,27,524 patches** of 50✕50 pixels were extract, that includes:
- **1,98,738** negative example (no Breast Cancer)
- **78,786** positive example (indicates Breast Cancer which were founded in the patches)

This clearly shows that we are taking class data 2x the num-bers of negatives data point than positive one’s. The image’s in the data set has a specified filename structure.
For example we take filename of a data set: **8863_idx5_x1101_y1151_class1.png**.

So we can Interpret this filename by:
- **Patient Id:** 8863_idx5
- **Coordinate of X:** 1101
- **Coordinate of Y:** 1151
- **Class labels:** 1 (Here 1 indicates IDC and 0 indicates no IDC)  

Above figure shows both positive and negative examples.
Our goal is to use this histology images to train a Deep Learn-ing model capable of telling the difference between the two class labels 1 and 0.

# PREPARING A MODEL FOR CANCER CLASSIFICATION

By using pip which is a python package manager, we will in-stall some of python Packages:

-	**Numpy:** supports large multidimensional matrices and array, along with high level mathematical function collection for operating these array.
-	**Pillow:** It is a python package which support wide ranges image file format’s and it is free for python programming language that add support’s for saving, manipulating and opening different format’s of image file.
-	**TenserFlow Keras:** Keras is a high level API for training and building Deep Learning model. It is used for production, state of the art research and prototyping.
-	**Imutils:** It is to make image processing function such as skeletonization, displaying, rotation, translation and resizing Matplotlib image as it is series of convenience function.
-	**Sckit-Learn** and **Matplotlip**

Applying these set of instructions to install all these packages using Terminal.

![image](https://user-images.githubusercontent.com/63160825/157169795-a7424a39-3a06-470b-8dc5-a28afd4bd231.png)

# PROJECT STRUCTURE:

Writing all the algorithms and scripts in python we save them all in a main directory named as **‘breast-cancer-classification’**. Inside this folder we created a new directory name as ‘**datasets**’ and inside datasets directory we created a new directory call ‘**orginal**’.

After arranging directories we will download the data set (‘Breast Histopathology Images’) from the official Kaggle 
![image](https://user-images.githubusercontent.com/63160825/157170038-166f5b34-339f-4aed-9ee5-47d92c255b89.png)
website (www.kaggle.com) by creating an account. We save the .zip file in breast-cancer-classificaton/datasets/orignal .
Now  we head back toward's terminal and navigating to the directory just created and unzip the file.

Now lets back to the main directory and use tree command to inspect the project structure: 
![image](https://user-images.githubusercontent.com/63160825/157170082-3494eccb-5bc3-4a18-aa9f-5d9b26a0bb23.png)
As we can see the data set in the **‘datasets/original’** folder and it is then divided into several faux patient’s ID. 

![image](https://user-images.githubusercontent.com/63160825/157170460-154d9f2c-d4ab-494d-87b1-f3da4b3e2709.png)
![image](https://user-images.githubusercontent.com/63160825/157170471-6d70a3fc-5d1c-471e-aade-1035945f345b.png)

Images in that patient’s ID is separated by **0 (benign)** and **1 (malignant)** directory, for eg:
			           
**9234** (Patient ID)
as 
**0**(images for benign)  **1**(images for malignant)

Here in this tree format the directory ‘cancernet’ contains our CancerNet and configuration.
Python files review:
- **config.py:** it is used by the model trainer and data set builder as it contains our configurations.
- **cancernet.py:** it contains CNN (Convolutional Neural Net-work) which is a classification of CancerNet breast cancer.
- **idc:** it is a directory that is going to store our data in the form of training, validation and testing. 
- **build_dataset.py:** it contains python script for splitting im-age’s into testing, validation and training. 
- **train_model.py:** this script uses TensorFlow- Keras for evaluating and training Breast Cancer classification.
- **.pyc files:** it contains compiled python file which has byte code of your source code.


# CONFIGURATION PROCESS

Before jumping into training our network and building our data set, let’s look over the configuration part of our classifier. 
I have created a file that stores all relevant configs which are actually python configurations which are necessary for the project to work efficiently, so please have a look at **‘config.py’**.
Starting from **Line1** importing the necessary package that is **os** stands for Operating System as this module provides a porta-ble way of OS-dependent function.

![image](https://user-images.githubusercontent.com/63160825/157170996-6a6052ce-5415-48a0-84e8-0f9b6345cc57.png)

Then in **Line3**, our config file is containing the path to the original data set (downloaded from Kaggle).

In **Line5** we are providing a base path where we are going to store our image after validation, testing and training split. 

From **Line6-8** we give deriving path to testing, validation and training output directories by using ‘BASE_PATH’

**Line10** we are assigning the percentage of the data that has to be used for the training by using ‘TRAIN_SPLIT’. We have set 80% here while rest of the 20% is used for testings.

In **Line11** we are reserving some of the data for validation. We are specify-ing 10% of the data set for validation (after splitting of the testing data).

Next we will be building image data set for Breast Cancer.

# BUILDING DATASET (BREAST CANCER)

The data set of Breast Cancer Image’s is of 1,98,783 image’s of 50✕50 pixels. This entire data needs to be approximately around 5.8 GB of hard disk storage. We organised our data set by scripting **‘build_dataset.py’**

![image](https://user-images.githubusercontent.com/63160825/157171109-008c5239-552d-468f-92a2-96e9e79c8646.png)

Again in **Line1-3**  we are importing **‘paths’** for collection of all image’s path and our ‘config’ configuration setting. We are al-so importing ‘random’ for shuffling randomly our path’s, **‘os’** for making directories and joining path’s and **‘shutil’**  for copy-ing images.

In **Line5-7**  we will be grabbing all the path’s of the images by **‘originalPaths’** for our data set and shuffling them using **‘shuffle.’**

![image](https://user-images.githubusercontent.com/63160825/157171223-9dd307bd-d340-4d3b-b246-e0812741f414.png)

**Line9-11** we are computing the path of testing and training, by using ‘**index**’. The ‘**trainPaths**’ and ‘**testPaths**’ which are generated by dividing the ‘**originalPaths**’.

Now in **Line13-15** we are splitting further the ‘**trainPaths**’, but this time we re-versing the portion for validation’s ‘**valPaths**’.

**Line17-19** a list is cre-ated with the name ‘**datasets**’ holding three tuple’s that con-tains required information of all our path ’**originalPaths**’ into validation, training and reserving data.

Next is looping the ‘datasets’ list:

![image](https://user-images.githubusercontent.com/63160825/157171583-ccc96bad-d12e-420b-afa3-609d9444deba.png)

On **Line22**  we define a loop over our dataset for splitting in which we:

**Line23**: shows the creation of data split

**Line25-27**: created a base Output Directories if it doesn’t exit then create it

**Line29**: created nested loop 	for all the input image in  the current split.	

**Line30-31**: extracting file name from the path and in line31 we are extracting label of the class for which the filename belongs.

**Line33**: building the path for the label directory

**Line34-36:** if the label output directory doesn’t exist then create it.

**Line38-39:** creating path to the destination image’s and then coping the image itself.

![image](https://user-images.githubusercontent.com/63160825/157171757-0c3c966f-c5cf-4b57-95d9-29e31db9b0b0.png)

When running our **‘build_dataset.py’** it creates our testing, validation and training directory structure by executing follow-ing command: 
 
I have inserted tree command also so that you can see the that the **‘dataset’** is now accurately structured into Validation, Training and Testing holding with class label’s 0 and 1 (Posi-tive and Negative Cancer Images).


# BREAST CANCER PREDICTION CNN-CancerNet

![image](https://user-images.githubusercontent.com/63160825/157171870-45ae4711-a053-4af6-8031-7cd9fec8ba13.png)
**FIG.Architecture for predicting Breast Cancer using Keras Deep Learning **

In the above figure we are showing the architecture is a Keras Deep Learning Classification for predicting  Breast cancer as it not such small architecture I have shown only a portion of it.

Now we are going to implement CNN architecture that we are going to use in this project. For implementing the CNN architecture we are going to use Deep Learning library name as Keras and along with that we have design a appro-priate network ‘**CancerNet**’ which:

- Uses 3✕3 CONV filter’s exclusive similar to VggNet
- Stack multiple 3✕3 CONV filter’s on top of each other to perform max-pooling (similar again to VggNet).
- Uses deep wise separable convolution rather than standard convolution layer’s.

Depth wise separable convolution is:

- More efficient.
- Less memory requirement.
- Less computation requires.
- Perform’s better then standard convolution in some of the situation.

Now lets check ‘**concert.py**’ :

![image](https://user-images.githubusercontent.com/63160825/157172553-a883d163-23ab-4594-8a9d-86fa1435bd9f.png)

From **Line1-9** here we are using Keras imports by that we will build CancerNet by using Keras sequential API’s. Here we are using imports :

- **SperacbleConv2D**: it allows depth-wise convolution’s as it is a type of a convolutional layer.
- **BatchNormalization**: it is used for the stability of Artificial Neural Network as well as for its performance and speed.
- **MaxPooling2D**: it is a discretisation process which is sam-ple based. Main objective is to down sample an input repre-sentation’s, reducing dimension’s.

**Line11-13**: we have defined a class CancerNet along with function build. As you can see we have add up four parame-ter’s in build function ‘**width, height, depth**’ that are specifying volume shape of the input image’s  to our network and depth is the colour image’s that each image is containing.
‘**Classes**’ will predict the number of Classe’s for our network as for ‘cancerNet’ it will be 2.

**Line14-16**: here we initialise ‘model’ by defining the ‘shape’. For using TenserFlow in backend we will now be able to add layers. 

**In Line18-20**: here we are specifying that if we are using ‘channels_first’ then we will be updating the in-put’s shape and channel’s dimension.


Now we will define **Depth-wise convolution => Relu =>Pool layers**:

![image](https://user-images.githubusercontent.com/63160825/157172823-07f285e9-398e-436b-9d7a-70715bb8cf48.png)

Here we have defined three block’s of **Depth-wise convolu-tion => Relu =>Pool** where we are increasing the number of filter’s and the stacking.

We have applied the ‘**BatchNormalization**’ and ‘**Dropout**’ as well. Dropout is a type of technique which is used for prevention from over fitting. As it work’s on random setting the out going edge’s of the hidden unit to 0 at each is updating for training phase.

Now appending our connected head:

![image](https://user-images.githubusercontent.com/63160825/157172912-dd5e248a-b17f-4426-b0aa-bf7daeaa80f0.png)

The FC => RELU layer’s and the ‘**softmax**’ classifier making the head of the network. 

The prediction percentage’s for each classes and model’s will be predicted by ‘**softmax**’ classifier as an output.

And at the end of this script we are returning it to the Training script.


# TRAINING THE KERAS BREAST CAN-CER MODEL

For training the model we have written a script of it **‘train_model.py’**: 

![image](https://user-images.githubusercontent.com/63160825/157173033-142423d6-43be-47a2-bf13-45584df87533.png)

Our imports are:

- **matplotlib**: it is a scientific plotting package that’s de facto standard for python. On Line2 we have define matplotlib to use the ‘Agg’ backend to that we are able to save our train-ing plot’s to disk.
- **keras**: we will be taking advantage’s of ‘Adagard’ optimiser, ‘np_utils’,’ImageDataGenerator’ and ‘Learn-ingRateSchedule’.
- **Sklearn**: We need implementations of ‘confusion_report’ and ‘classification_report’ from scikit-learn.
- **Imutils** and **numpy**

![image](https://user-images.githubusercontent.com/63160825/157173151-d38a368c-37a7-4595-85e5-c014c2cf2439.png)

In **Line17** we are defining baths size that is assigned 32 then initial learning rate by 1e-2 and last one number’s of training epoch’s that is 40.

From **Line19-22** we will be grabbing our training image’s path and determining the total number’s of image in each one of the split. Then in **Line24-27**  we will be computing the **‘classweight’** for our training-data for ac-counting class skew or imbalance.

Now we will be initialising the training data augmentation ob-ject:

![image](https://user-images.githubusercontent.com/63160825/157173751-8855ac2c-e417-45df-b982-dd57fd61b5f3.png)

Data argumentation is important to all of the Deep Learning experiment’s for assisting with model generalisation and it is a form of regularisation.

Here we have initialised ‘trainAug’ which is a data augmentation object’s from **Line29-38**. Shifts, shear’s , flip’s and random rotation’s are applied to the data when it will be generated. Image pixel’s intensitie’s from range {0,1} is rescaled by ‘trainAug’ generator. 

In **Line40** testing and validation data argumentation object is initialised. Now we will initialise our generator now:

![image](https://user-images.githubusercontent.com/63160825/157173850-53a383ae-cd7a-45c2-a7b6-05a0e823da53.png)

Here we are initialising the testing, validation and training generators. Batches of Image’s on demand will be provided by each of the generators and it is donated by the parameter ‘**batch_size**’.

Now we will initialise the model and start the training process:

![image](https://user-images.githubusercontent.com/63160825/157173919-9d03fe32-c6db-4ada-bfa0-d4414fdabbd6.png)

In **Line64-65** we have initialised the model by ‘Adagrad’ opti-miser. Then the compilation of the model take place and we use ‘compile’ to compile our model with ‘bina-ry_crossentropy’ by ‘loss’ function.

By calling ‘**fit_generator**’ our training process’s is initiated, by using this method rather than having the whole data set in Random Access Memory thought training, we have our image data set in our disk.

After completion of training, model is evaluated on the our testing data.

![image](https://user-images.githubusercontent.com/63160825/157174364-71b606b5-3777-4155-8736-be21e095d75f.png)

In **Line79** we are making prediction’s on all of the testing data by using generator object again. And in  **Line83** we are print-ing the classification report by using ‘**classification_report**’  to the terminal.

Gathering additional evaluation metric’s:

![image](https://user-images.githubusercontent.com/63160825/157174469-85396599-c1db-400a-bce9-7a903153c532.png)

Here by computing the ‘**confusion_matrix**’ and then deriving the accuracy ‘**accuracy**’, specificity ‘**specificity**’ and sensitivity ‘**sensitivity**’ in **Line85-89**. In **Line90-93** the values and the matrix is printed in the terminal. 

Storing and generating training plot:

![image](https://user-images.githubusercontent.com/63160825/157174569-fef23d6b-f82d-422e-b931-75188d4e8943.png)

The training history plot consist’s of training/validation accura-cy and training/validation loss. Plotted over time we can spot under-fitting/over.

# Result of prediction of Breast Cancer

Now we will run our final script ‘**train_model.py**’ in terminal:

![image](https://user-images.githubusercontent.com/63160825/157175046-f2d74f35-6558-4b22-bb85-57b83749a38e.png)

As in the above Figure of terminal you can see that we have executed our ‘**train_model.py**’ script in the terminal and train-ing of our model started taking place. I was using my Mac-Book Air in this whole project and it has taken allot of time for training this model as you can see in above terminal image that every approach is going for about 2000s 300ms/step which is very long time taking according to such amount of data set and whole lot there were about 40 total approaches to take place and it has taken almost 3 hours to complete 1 to 4 approaches and about 85% of the 5th one so you can as-sume how much time it will take, approximately 12 hours to complete. 

![image](https://user-images.githubusercontent.com/63160825/157175065-1251f837-ab5e-4e44-92ac-cab827a973ae.png)

Output that our model has achieved is for about 85% accura-cy, while the raw accuracy is heavily weighted that is why it is classified no-cancer/benign correctly for 93% for all of the time.

![image](https://user-images.githubusercontent.com/63160825/157175097-fbf5a7cc-8a3f-49d7-b713-4fbe5c1c122b.png)

The above graph is a our CancerNet classification model training plot generated with Keras.

For understanding our model performance in the deepest level compute the specificity and sensitivity. Our sensitivity is the true positive’s that we also have predicted as positive for about 85.03%. Measurement of specificity is our true negative about 84.70%. 

**We have to be very careful with our negative false here as we did not want to classify that someone as No cancer when there are positive one. Same wise our positive false is also important as we did not want someone to be Cancer positive and then subjected them to expensive, in-vasive and painful treatment when they did not need them.**	
	      
**Always there should be balance between specificity and sensitivity that’s a Deep learning or Machine Learning Engineer should manage and when coming to the Health Care and medical facility it become more important.**

The challenging data set along with class imbalance lead’s for obtaining **~85% specificity**, **~86% classification** accuracy and **~85% sensitivity**.







