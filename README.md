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

Basically in this version of project we build a classifier for training a breast cancer histology data set from about 80% data set. Beside all this we have taken almost 10% of data set for the validation purpose. We have defined CNN (Convolutional Neural Network) using Keras we call it CancerNet and our images in the data sets are trained by it. The model performance is analysed by deriving confusion ma-trix.
