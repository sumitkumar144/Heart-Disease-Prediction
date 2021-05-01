# Heart-Disease-Prediction Using Machine Learning

## Table of Content
  * [Demo](#demo)
  * [Motivation](#motivation)
  * [Overview](#overview)
  * [System Configuration](#System-Configuration)
  * [Software Used](#Software-Used)
  * [Programming Language Used](#Programming-Language-Used )
  * [Python Library Used](#Python-Library-Used )
  * [Dataset](#Dataset)
  * [Methodology](#Methodology)
  * [Evaluation](#Evaluation)
  * [Discussion and Conclusion](#Discussion-and-Conclusion)
  * [Deployment](#deployment)
  * [Future Scope](#Future-Scope)
  * [Credits](#credits)
  
  ## Demo
   
  ![Alt text](static/img/heart-disease-prediction.gif)

 ## Motivation
 
 Coronary heart disease (CHD) also known as heart disease or coronary artery disease or cardiovascular disease is a major cause of death across worldwide. About 17.9 million people in 2016 were died because of CHD, which is 31% of all global deaths. Low- and medium-income countries contributed more than three quarter of these deaths. Coronary heart disease occurs mostly because of the deposit of fatty substances in the coronary arteries of heart and that in-tern stops the blood supply in heart. The major behavioural risk factors are unhealthy diet, obesity, gender, age, cholesterol, high blood pressure, and family history, use of tobacco, hypertension, diabetes, excessive and harmful use of tobacco, air pollution and physical inactivity. The main symptoms of a coronary heart disease are chest paint (angina), shortness of breath, pain throughout the body, feeling faint, feeling sick (nausea). But everyone might not have similar symptoms and some people might not have any of these symptoms before they are diagnosed. Risk of getting a coronary heart disease can be reduced by making few behavioural changes like being physically active, eating healthy and balanced diet, controlling blood cholesterol and sugar levels, and giving up smoking. British Heart Federation reported that, approximately 5000 more people in England have died from heart disease between the start of pandemic in March 2020 and Mid October 2020. A  number of factors like people fearing putting off excess pressure on NHS by seeking care, people delaying contacting NHS because of the fear of getting infected by COVID-19, delay to routine heart care and surgeries because of outnumbered patients from COVID-19 could have led to the excess deaths.
The aim of this project is to mitigate the risk of this excessive death arisen due to Coronary heart disease during pandemic and help NHS better forecast the number of patients affected by coronary heart disease, it is very important to predict them using cutting edge technology like machine learning by making use of clinical data available in the healthcare sector. Predicting CHD though machine learning is a binary classification problem. Since it involves high risk of dying of CHD because of incorrect prediction, it is very important to predict the chances of a person having heart disease more accurately. Many research topics in the past have tried to design and develop a better and accurate system for predicting coronary heart disease.

This is purely for research, education and informational purpose.

## Overview

Heart Disease Prediction is a simple yet important and critical binary machine learning classification problem. Hence it is very important to increase the accuracy of classification and decresing the misclassification of presence of heart disease.

This project was done by collecting the dataset from [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci) which in turn taken from [UCI machin learning repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) for cleveland clinic foundation. then analysing the data by using several exploratory data analysis techniques to identify keep observation on data which is mentioned in detail in exploratory data analysis section. 
Based on the identified observation, feature engineering was done on the data to remove outliers from the numerical features. Then feature selection technique was performed on the data to identify corelated features and remove them from the original dataset. There was no significant Correlation between the numerical data. Hence, no features were removed from the dataset. 
The pre-processed dataset was split using holdout method with 80:20 ratio for training and testing dataset respectively. Then the training and testing datasets were scaled down separately to make mean of zero and standard deviation of one as some of the machine learning algorithms like logistic regression assume the sample to be normally distributed. 
Once the pre-processing was done on the data the base classifier models (logistic regression, decision tree, K-Nearest Neighbour, and artificial neural network) were trained and tested using those data. Then hyperparameter optimization technique was used to identify the best hyperparameters. While applying hyperparameter optimization technique 10-fold cross validation is used to generalise the learning patter of the models. The base classifier models were trained again using those best hyperparameters. 
Then machine learning algorithms like bagging and stacking with hyperparameter optimization technique was used to train the models again. While performing hyperparameter optimization of bagging and stacking classifier 5-fold cross validation is used. After all the experiments were done, performance of the models was evaluated by using the performance metrics best suited for heart disease dataset. Then the model with higher performance is selected for further prediction of coronary heart disease when a new input data is provided to the model. 
The selected model was then deployed into Heroku as well as AWS cloud along with the beautifully designed, user reliable and convenient web interface so that user input can be provided to the model through the web interface. Based on the input provided by the user, prediction was made and displayed to the user again.

## Dataset

<code>heart.csv</code>, collected from the [Kaggle Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) challenge, contains 14 biological attributes of 303 people, including whether the person has heart disease or not.

### Biological Attributes

Features used for training machine learning models on, including the special binary class label <b><i>target</b></i>, describing whether heart disease was detected.

1. <b><i>age</i></b>: Age in years
2. <b><i>ca</i></b>: Number of major blood vessels (0-3)
3. <b><i>chol</i></b>: Serum cholestrol in mg/dl
4. <b><i>cp</i></b>: Chest pain type
    * Value 1: Typical angina
    * Value 2: Atypical angina
    * Value 3: Non-anginal pain
    * Value 4: Asymptomatic
5. <b><i>exang</i></b>: Exercise induced angina (1 = yes; 0 = no)
6. <b><i>fbs</i></b>: fasting blood sugar > 120 mg/dl (1 = true; 0 = no)
7. <b><i>oldpeak</i></b>: ST depression induced by exercise relative to rest
8. <b><i>restecg</i></b>: Resting electrocardiographic results
    * Value 0: Normal
    * Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
9. <b><i>sex</i></b>: Sex (1 = male; 0 = female)
10. <b><i>slope</i></b>: The slope of the peak exercise ST segment
    * Value 1: Upsloping
    * Value 2: Flat
    * Value 3: Downsloping
11. <b><i>target</i></b>: Heart disease detection (0 = disease; 1 = no disease)
12. <b><i>thal</i></b>: Thalium stress test
    * Value 3: normal
    * Value 6: fixed defect
    * Value 7: reversibe defect
13. <b><i>thalach</i></b>: Maximum heart rate achieved in bpm
14. <b><i>trestbps</i></b>: Resting blood pressure (in mmHg on admission to the hospital)

## System Configuration

| Name | Specification |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Processor | Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz, 4 Core |
| SSD | 256GB |
| RAM | 8GB |
| GPU | Intel(R) UHD Graphics 620, 4GB |


## Software Used

| Name | Version |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Anaconda Distribution | ![Anaconda Navigator 1.10.0](https://img.shields.io/badge/Anaconda%20Navigator-1.10.0-orange) |
| PyCharm | ![PyCharm 2021.1](https://img.shields.io/badge/PyCharm-2021.1-blue) |
| Visual Studio Code | ![Visual Studio Code 1.55.2](https://img.shields.io/badge/Visual%20Studio%20Code-1.55.2-yellowgreen) |


## Programming Language Used

| Name | Version |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python | ![Python 3.8](https://img.shields.io/badge/python-3.8-green) |
| HTML | ![HTML 5](https://img.shields.io/badge/HTML-5-lightgrey) |
| CSS | ![CSS 3](https://img.shields.io/badge/CSS-3-red) |
| JavaScript | ![JavaScript ES6](https://img.shields.io/badge/JavaScript-ES6-brightgreen) |


## Python Library Used

| Name | Version |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scikit-learn | ![scikit-learn 0.24.1](https://img.shields.io/badge/scikit--learn-0.24.1-brightgreen)|
| tensorflow | ![tensorflow 2.2.0](https://img.shields.io/badge/tensorflow-2.2.0-green)|
| pandas | ![pandas 1.2.4](https://img.shields.io/badge/pandas-1.2.4-yellowgreen) |
| numpy | ![numpy 1.20.2](https://img.shields.io/badge/numpy-1.20.2-yellow) |
| matplotlib | ![matplotlib 3.4.1](https://img.shields.io/badge/matplotlib-3.4.1-orange) |
| seaborn | ![seaborn 0.11.1](https://img.shields.io/badge/seaborn-0.11.1-red) |
| Flask | ![Flask 1.1.2](https://img.shields.io/badge/Flask-1.1.2-lightgrey) |
| Flask-Cors | ![Flask-Cors 3.0.10](https://img.shields.io/badge/Flask--Cors-3.0.10-blue) |

## Methodology

### Exploratory Data Analysis

Exploratory data analysis is an important statistical approach in the life cycle of machine learning to analyse dataset to find out patterns, missing values, outliers etc. using visualisation, inbuilt pandas function. So that a feature engineering can be performed on them at later point of time to make it suitable for the machine learning models. Key Observation:
 - None of the columns are having missing values.
 - All the attributes are in numeric form.
 - The magnitude of range of values for attributes age, blood pressure, serum cholesterol, max heart rate and thalach are high with comparison to the other attributes.
 - From the dataset description, it was observed that most of the features are categorical.
 
### Feature Engineering

This dataset did not require much of feature engineering as most of the features are already in a form where they are suitable for machine learning models. However the numerical features were treated for outlier to bring the outliers values to either the upper extreme value where outliers are greater than the upper extreme value in a box plot or with lower extreme value where outliers are lesser than the lower extreme value in a box plot.

### Feature Selection

As a part of feature selection heatmap and variance inflation factor is used to perform colinearilty and multicolinearity check of numerical features respectively. So that highly corelated features can be removed. However no significant observation were made.

### Chossing Base Classifier

4 base classifier were chose for model training. These models were chosen because of the differfence in their learning pattern.
- Logistic Regression
- Dicession Tree
- K-Nearest Neighbour
- Artificial Neural Network
After training the model with above base classifier the performance of the models were noted.

### Hyperparameter Optimization for Base Classifier

10-fold cross validation along with Grid-Search CV is used to identify the best hyperparameters for the above base classifier. After identifying the base hyperparameters the base classifiers were trained again and the performance were noted. 

### Bagging Classifier with Hyperparameter Optimization

Bagging classifier with all the above base classifier were used along with Grid-Search CV to indentify the best hyperparameter. Then the bagging classifiers for each of the base classifiers were trained again and performance of the models were noted down.

### Stacking Classifier with Hyperparameter Optimization

Stacking classifier using Decision Tree, K-Nearest Neighbour as base classifier and Logistic Regression as Meta Classifier were trained using Grid-Search CV to indentify the best hyperparameters. After identifying the best hyperparameters the stacking classifier was trained again and performance were noted.


## Evaluation

### Evaluation of Base Classifier

| Base Classifier | Accuracy | Precision | Recall | F1-Score | AUC | ROC |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Logistic Regression | 81.96 | 89.65 | 76.47 | 82.53 | 82.32 | 82.00 |
| Decision Tree | 80.32 | 79.31 | 79.31 | 79.31 | 80.28 | 80.00 |
| **K-Nearest Neighbour** | **88.52** | **93.10** | **84.37** | **88.53** | **88.73** | **89.00** |
| Artificial Neural Network | 85.24 | 89.65 | 81.25 | 85.24 | 85.45 | 85.00 |

### Evaluation of Base Classifier after hyperparameter optimization

| Base Classifier with CV | Accuracy | Precision | Recall | F1-Score | AUC | ROC |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Logistic Regression | 83.60 | 89.65 | 78.78 | 83.87 | 83.89 | 84.00 |
| Decision Tree | 85.24 | 89.65 | 81.25 | 85.24 | 85.45 | 85.00 |
| **K-Nearest Neighbour** | **88.52** | **93.10** | **84.37** | **88.53** | **88.73** | **89.00** |

### Performance of Bagging classifier

| Bagging Classifier with CV | Accuracy | Precision | Recall | F1-Score | AUC | ROC |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Logistic Regression | 88.52 | 86.20 | 89.28 | 87.71 | 88.41 | 88.00 |
| **Decision Tree** | **91.80** | **86.20** | **95.15** | **90.90** | **91.54** | **92.00** |
| K-Nearest Neighbour | 85.24 | 93.10 | 79.41 | 85.71 | 85.61 | 86.00 |

### Performance of Stacking classifier

| Stacking Classifier with CV | Accuracy | Precision | Recall | F1-Score | AUC | ROC |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| DT, KNN as base model and LR as Meta Model | 88.52 | 93.10 | 84.37 | 88.52 | 88.73 | 89.00 |


## Discussion and Conclusion

From the evaluation below observation were made
- Among the base classifier K-Nearest Neighbour with 7 Neighbour outperformed other classifiers.
- After the hyperarameter optimization the performance of each of the base classifier increased and K-Nearest Neighbour is still highest performing model with 9 neighbours.
- Among all the bagging classifiers Decision Tree with 62 Decision Tree outperformed other bagging classifiers.
- Stacking classifier performance is equivalent to K-Nearest Neighbour base classifier. 

Hance, Bagging Classifier built with 62 Decision Tree is chosen as the final model for prediction.

## Deployment

### Deployment into Heroku

 Heroku Deployment Link: [https://prediction-heart-disease.herokuapp.com/](https://prediction-heart-disease.herokuapp.com/)

### Deployment into AWS EC2

AWS Deployment Link: [http://ec2-3-21-56-244.us-east-2.compute.amazonaws.com:8080](http://ec2-3-21-56-244.us-east-2.compute.amazonaws.com:8080)

#### AWS Specification
 - **Server:** Ubuntu Server 20.04 LTS (HVM), SSD Volume Type - ami-08962a4068733a2b6 (64-bit x86) / ami-064446ad1d755489e (64-bit Arm)
 - **Instance Type:**   t2.micro (- ECUs, 1 vCPUs, 2.5 GHz, -, 1 GiB memory, SSD 8GB)

## Future Scope
Each time an input data is given for prediction, should be collected in the database and then model should be retrained at regular interval to increase the generalization capabi;ity of the model.

## Credits
### Coding Credit
- [Krish Naik](https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig)
- [Josh Starmer](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
- [iNeuron iNtelligence](https://www.youtube.com/channel/UCb1GdqUqArXMQ3RS86lqqOw)
- [Jason Brownlee](https://machinelearningmastery.com/)
- [Adrian Kochanski](https://github.com/kochansky/heart-disease-prediction/)
- [G. Shreekant](https://github.com/g-shreekant/Heart-Disease-Prediction-using-Machine-Learning)
- [Shalaka Saraogi](https://github.com/shalakasaraogi/heart-disease-prediction/)
