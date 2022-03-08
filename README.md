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


