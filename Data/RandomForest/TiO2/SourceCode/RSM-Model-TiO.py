#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
np.random.seed(500)
import tensorflow as tf
tf.set_random_seed(400)
import pandas as pd
import matplotlib.pyplot as plt

import pickle as cPickle

from openpyxl import load_workbook

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 12,
        }

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from matplotlib.legend_handler import HandlerLine2D


# In[7]:


# LoadData set function definition
def loadTrainingData(path, fileName):
    """
    fileName: the fileName is considered as excel
    
    """
    # value is hard coded, has to implement the fileName and path style declaration
    file = pd.ExcelFile(r'./Data/Data-RSM.xlsx')
    data = pd.read_excel(file, "Data-Final")
    #check if the dataFrame is empty
    if data.empty:
        print("Load Data Error: Data not found in path")
        return
    # Copy the original data, just to avoid the change in original data
    dataCopy = data.copy()
    #shuffling of the obtained dataFrame
    dataCopy = dataCopy.sample(frac=1).reset_index(drop=True)
    # accessing the values of the trainingSetColumns
    inputFeaturesDF = dataCopy.iloc[:,0:4]
    # changing trainingDF to Numpy array
    inputFeatures = inputFeaturesDF.values    
     # accessing the targetColumns
    targetLabels = dataCopy.columns[-1]
     # accessing the targetColumns
    targetLabelsDF = dataCopy.loc[:,targetLabels]
    # changing targetDF to Numpy array
    targetLabels = targetLabelsDF.values
    #return the values 
    return (inputFeatures, targetLabels, dataCopy)


# In[8]:


#Load the data
trainingInputFeatures, trainingTargetLabels, datacopy = loadTrainingData('./Data','/Data-RSM.xlsx')

#printing the dimensions of the data
print("Dimensions: TrainingFeatures Data shape is:", trainingInputFeatures.shape)
print("Dimensions: TrainigLabels Data shape is:", trainingTargetLabels.shape)


# In[9]:


# LoadData set function definition
def loadExperimentData(path, fileName):
    """
    fileName: the fileName is considered as excel
    
    """
    # value is hard coded, has to implement the fileName and path style declaration
    file = pd.ExcelFile(r'./Data/AI-Design-ML-CC-TiO-Experiment.xlsx')
    data = pd.read_excel(file, "TiO-Experiment")
    #check if the dataFrame is empty
    if data.empty:
        print("Load Data Error: Data not found in path")
        return
    # Copy the original data, just to avoid the change in original data
    dataCopy = data.copy()
    #shuffling of the obtained dataFrame
    dataCopy = dataCopy.sample(frac=1).reset_index(drop=True)
    # accessing the values of the trainingSetColumns
    inputFeaturesDF = dataCopy.iloc[:,0:4]
    # changing trainingDF to Numpy array
    inputFeatures = inputFeaturesDF.values    
     # accessing the targetColumns
    targetLabels = dataCopy.columns[-1]
     # accessing the targetColumns
    targetLabelsDF = dataCopy.loc[:,targetLabels]
    # changing targetDF to Numpy array
    targetLabels = targetLabelsDF.values
    #return the values 
    return (inputFeatures, targetLabels, dataCopy)


# In[10]:


#Load the data
experimentInputFeatures, experimentTargetLabels, datacopy = loadExperimentData('./Data','/AI-Design-ML-CC-TiO-Experiment.xlsx')

#printing the dimensions of the data
print("Dimensions: TrainingFeatures Data shape is:", experimentInputFeatures.shape)
print("Dimensions: TrainigLabels Data shape is:", experimentTargetLabels.shape)


# In[11]:


def normaliseData (features, labels):
    """
    Function to scale the given X, Y
    X: Input data, scaled using the sklearn preprocessing
    Y: Ouptu data, scaled using the sklearn preprocessing
    """
    # access the scalar for input
    scalerX = MinMaxScaler()
    # fit the data to the scaler to scale the input data set 
    scalerX = scalerX.fit(features)
    # transform it to make the input normalised
    featuresScaled = scalerX.transform(features)
    # reshaping the targetY as it has only one feature or label
    labels = labels.reshape(-1, 1)
    # access the scalar for output
    scalerY = MinMaxScaler()
    # fit the data to the scaler to scale the output data set 
    scalerY = scalerY.fit(labels)
    # transform it to make the output normalised
    labelsScaled = scalerY.transform(labels)
    return featuresScaled, labelsScaled


# In[12]:


# scale data
trainingInputFeaturesScaled, trainingTargetLabelsScaled = normaliseData(trainingInputFeatures, trainingTargetLabels)
print("Dimensions: Training DataScaled Shape:", trainingInputFeaturesScaled.shape)
print("Dimensions: Target DataScaled Shape:", trainingTargetLabelsScaled.shape)


# In[13]:


# scale data
experimentInputFeaturesScaled, experimentTargetLabelsScaled = normaliseData(experimentInputFeatures, experimentTargetLabels)
print("Dimensions: Training DataScaled Shape:", experimentInputFeaturesScaled.shape)
print("Dimensions: Target DataScaled Shape:", experimentTargetLabelsScaled.shape)


# In[14]:


def splitData (featuresScaled, labelsScaled):
    """
    Split the data using the sklearn preprocessing tool
    Splitting will be done two times, to get training data, evaluation data, test data
    trainXScale: the parameter where the scaled input is fed to be split as 70, 15, 15 (train, evaluate, test)
    targetYScale: The parameter where the scaled output is fed to be split as 70, 15, 15
    
    """
    
    # use the sklearn split function to split, the sklearn gives only 
    trainingFeatures, trainingFeaturesValAndTest, targetLabels, targetLabelsScaledValAndTest = train_test_split(featuresScaled, labelsScaled, test_size=0.3, random_state=42)
    
    # We are again splitting the data to bring our evaluation set
    trainingFeaturesVal, testFeatures, targetLabelsVal, testLabels = train_test_split(trainingFeaturesValAndTest, targetLabelsScaledValAndTest, test_size=0.5, random_state=42)
    
    # printing the shape of the vectors
    #print(trainX.shape, trainXEval.shape, trainXTest.shape, targetY.shape, targetYEval.shape, targetYTest.shape)
    
    return trainingFeatures, trainingFeaturesVal, testFeatures, targetLabels, targetLabelsVal, testLabels


# In[15]:


# split data
trainingFeatures, trainingFeaturesVal, testFeatures, targetLabels, targetLabelsVal, testLabels = splitData(trainingInputFeaturesScaled, trainingTargetLabelsScaled)

# printing the shapre of the data for confirmation

print("Dimensions: Training Dataset Shape:", trainingFeatures.shape)
print("Dimensions: Training Target Dataset Shape:", targetLabels.shape)

print("Dimensions: Evaluation Training Features Dataset Shape:", trainingFeaturesVal.shape)
print("Dimensions: Evaluation Target Dataset Shape:", targetLabelsVal.shape)

print("Dimensions: Test Features Dataset Shape:", testFeatures.shape)
print("Dimensions: Test Target Dataset Shape:", testLabels.shape)


#concatenate the experiment testing data to the training data

experimentInputFeaturesNew = np.concatenate((experimentInputFeaturesScaled,testFeatures))
print("Dimensions: Experiment Input Features Dataset Shape:", experimentInputFeaturesNew.shape)


experimentTargetLabelsNew = np.concatenate((experimentTargetLabelsScaled,testLabels))
print("Dimensions: Experiment Target Labels Dataset Shape:", experimentTargetLabelsNew.shape)


# In[16]:


#Prediction Function Definition
def predictUsing(model, test_features, test_labels):
    # predict the values for the given input using mlModel
    predictions = model.predict(test_features)
    #calculate the mean_square_error
    mse = mean_squared_error(predictions, test_labels)
    # compute the difference between the *predicted* CC and the *actual* CC
    errors = abs(predictions - test_labels)
    return mse, predictions


# In[17]:


def writeMLPModelPredictionsToExcel(test_labels, predicted_labels):
    df = pd.DataFrame({'Target':test_labels,'Predicted': predicted_labels})
    df.to_excel('./Data/AI-Design-ML-CC-Experiment-TiO-MLPModel.xlsx', sheet_name='MLPModel', index=False)
    return


# In[18]:


def writeRFModelPredictionsToExcel(test_labels, predicted_labels):
    df = pd.DataFrame({'Target':test_labels,'Predicted': predicted_labels})
    df.to_excel('./Data/AI-Design-ML-CC-Experiment-TiO-RFModel.xlsx', sheet_name='RFModel', index=False)
    return


# In[19]:


def drawErrorHistogram(predictions, target_labels):
    #printing the error
    error = predictions - target_labels
    plt.hist(error,color="r", bins=25, rwidth=0.8, range=[-0.2,0.2], align='mid')
    plt.xlabel("Prediction Error", fontdict=font)
    plt.ylabel("Occurrences", fontdict=font)
    return


# In[20]:


def loadMLPmodel(mlpModelName):
    savedMLPModel = load_model(mlpModelName)
    #check if MLPModel exists or not
    if savedMLPModel is None:
        print("No saved MLPModel with the name:",mlpModelName)
        return
    #print savedModel summary
    print(savedMLPModel.summary())
    return savedMLPModel


# In[21]:


def loadRFmodel(rfModelName):
    savedRFModel = cPickle.load(open(rfModelName, 'rb'))
    #check if RFModel exists or not
    if savedRFModel is None:
        print("No saved RFModel with the name:",rfModelName)
        return
    return savedRFModel


# In[22]:


#load saved model
savedMLPModel = loadMLPmodel("model-L-1-N-10-N-5.h5")
mse, predictions = predictUsing(savedMLPModel, experimentInputFeaturesNew, experimentTargetLabelsNew)
#print the MSE
print("MSE: ", mse)
#Flattern the prediction and testLabels
mlp_predictions = predictions.flatten()
mlp_testLabels = experimentTargetLabelsNew.flatten()
#draw error histogram
drawErrorHistogram(mlp_predictions, mlp_testLabels)


# In[23]:


#Write the predictions of mlpModel to the excel sheet
writeMLPModelPredictionsToExcel(mlp_testLabels, mlp_predictions)


# In[24]:


#load saved RFmodel
savedRFModel = loadRFmodel("RFModel")
#use the predict function to predict the targetLabels
mse, predictions = predictUsing(savedRFModel, experimentInputFeaturesNew, experimentTargetLabelsNew.ravel())
#flatten the predictions array
rf_predictions = predictions.flatten()
rf_testLabels = experimentTargetLabelsNew.flatten()
#plot the error histogram
drawErrorHistogram(rf_predictions, rf_testLabels)
print("MSE: ", mse)


# In[25]:


#Write the predicted values to excel sheet
writeRFModelPredictionsToExcel(rf_testLabels, rf_predictions)


# In[ ]:




