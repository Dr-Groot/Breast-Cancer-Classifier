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
As we can see the data set in the **‘datasets/original’** folder and it is then divided into several faux patient’s ID. Images in that patient’s ID is separated by **0 (benign)** and **1 (malignant)** directory, for eg:
			           
**9234** (Patient ID)
as 
**0**(images for benign)  **1**(images for malignant)




