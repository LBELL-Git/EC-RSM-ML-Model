#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary libraries to build the RandomForest

#framework to handle the data read from excel or SVS

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.25

from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 12,
        }

import pickle as cPickle
import seaborn as sns
import numpy as np
import pydot


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib.legend_handler import HandlerLine2D
from sklearn.model_selection import GridSearchCV


# In[2]:


# LoadData function definition

def loadData(path, fileName):
    """
    fileName: the fileName is considered as excel
    
    """
    # value is hard coded, has to implement the fileName and path style declaration
    file = pd.ExcelFile(r'./Data/RSM-Data.xlsx')
    data = pd.read_excel(file, "Data-Final")
    
    #check if the dataFrame is empty
    if data.empty:
        print("Load Data Error: Data not found in path")
        return
    
    # Copy the original data, just to avoid the change in original data
    dataCopy = data.copy()
    
    #shuffling of the obtained dataFrame
    dataCopy = dataCopy.sample(frac=1).reset_index(drop=True)
    
    #printing the dimensions of the data
    print("Dimensions: Total Dataset shape:", dataCopy.shape)

    # accessing the values of the trainingSetColumns
    trainingData = dataCopy.copy()
    trainingDataDF = trainingData.iloc[:,0:4]
    trainingFeaturesNames = list(trainingDataDF.columns)
    
    # changing trainingDF to Numpy array
    trainingFeatures = trainingDataDF.values
    
     # accessing the targetColumns
    targetDataColumn = dataCopy.columns[-1]
    
     # accessing the targetColumns
    targetDataDF = dataCopy.loc[:,targetDataColumn]
    
    # changing targetDF to Numpy array
    targetLabels = targetDataDF.values
    
    #return the values 
    return (trainingFeaturesNames, trainingFeatures, targetLabels, dataCopy)


# In[3]:


#Load the data
trainingFeaturesNames, trainingFeatures, targetLabels, datacopy = loadData('./Data','/RSM-Data.xlsx')

print("Training Features: ", trainingFeaturesNames)
print("Dimensions: Training Data Shape:", trainingFeatures.shape)
print("Dimensions: Target Data Shape:", targetLabels.shape)


# In[4]:


def normaliseData (trainingFeatures, targetLabels):
    """
    Function to scale the given trainingFeatures, targetLabels
    trainingFeatures: Input data, scaled using the sklearn preprocessing
    targetLabels: Ouptu data, scaled using the sklearn preprocessing
    """
    # access the scalar for input
    scalerTrainingFeatures = MinMaxScaler()
    # fit the data to the scaler to scale the input data set 
    scalerTrainingFeatures = scalerTrainingFeatures.fit(trainingFeatures)
    # transform it to make the input normalised
    trainingFeaturesScaled = scalerTrainingFeatures.transform(trainingFeatures)
    
    # reshaping the targetY as it has only one feature or label
    targetLabels = targetLabels.reshape(-1, 1)
    
    # access the scalar for output
    scalerTargetLabels = MinMaxScaler()
    # fit the data to the scaler to scale the output data set 
    scalerTargetLabels = scalerTargetLabels.fit(targetLabels)
    # transform it to make the output normalised
    targetLabelsScaled = scalerTargetLabels.transform(targetLabels)
    return trainingFeaturesScaled, targetLabelsScaled


# In[5]:


# scale data
trainingFeaturesScaled, targetLabelsScaled = normaliseData(trainingFeatures, targetLabels)

print("Training Features: ", trainingFeaturesNames)
print("Dimensions: Training DataScaled Shape:", trainingFeaturesScaled.shape)
print("Dimensions: Target DataScaled Shape:", targetLabelsScaled.shape)


# In[6]:


def splitData (trainingFeaturesScaled, targetLabelsScaled):
    """
    Split the data using the sklearn preprocessing tool
    Splitting will be done two times, to get training data, evaluation data, test data
    trainingFeaturesScaled: the parameter where the scaled input is fed to be split as 70, 15, 15 (train, evaluate, test)
    targetFeatureScaled: The parameter where the scaled output is fed to be split as 70, 15, 15
    
    """
    
  # use the sklearn split function to split, the sklearn gives only 
    trainingFeatures, trainingFeaturesValAndTest, targetLabels, targetLabelsScaledValAndTest = train_test_split(trainingFeaturesScaled, targetLabelsScaled, test_size=0.3, random_state=42)
    
    # We are again splitting the data to bring our evaluation set
    trainingEvaluationFeatures, testFeatures, targetEvaluationLabels, targetTestLabels = train_test_split(trainingFeaturesValAndTest, targetLabelsScaledValAndTest, test_size=0.5, random_state=42)
  
    #return trainingFeatures, trainingEvaluationFeatures, testFeatures, targetLabels, targetEvaluationLabels, targetTestLabels
    return trainingFeatures, testFeatures, targetLabels, targetTestLabels


# In[7]:


# split data

#trainingFeatures, trainingEvaluationFeatures, testFeatures, targetLabels, targetEvaluationLabels, targetTestLabels = splitData(trainingFeaturesScaled, targetLabelsScaled)

trainingFeatures, testFeatures, targetLabels, targetTestLabels = splitData(trainingFeaturesScaled, targetLabelsScaled)

print("Dimensions: Training Features Dataset Shape:", trainingFeatures.shape)
print("Dimensions: Training Target Dataset Shape:", targetLabels.shape)

print("Dimensions: Test Features Dataset Shape:", testFeatures.shape)
print("Dimensions: Test Target Dataset Shape:", targetTestLabels.shape)


# In[8]:


def loadRFmodel(rfModelName):
    loaded_model = cPickle.load(open(rfModelName, 'rb'))
    return loaded_model


# In[9]:


#Prediction Function Definition
def predictUsing(rfModel, features, labels):
    # Use the forest's predict method on the test data
    predictions = rfModel.predict(features)
    #calculate the mean_square_error
    mse = mean_squared_error(predictions, labels)
    # Calculate the absolute errors
    mae = mean_absolute_error(predictions, labels)
    return rfModel,mse, predictions


# In[10]:


#visualise the importances
def showImportantParametersUsing(rfModel,trainingFeaturesNames):
    
    importances = list(rfModel.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(trainingFeaturesNames, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # Import matplotlib for plotting and use magic command for Jupyter Notebooks
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')
    # Tick labels for x axis
    plt.xticks(x_values, trainingFeaturesNames, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
    return
    


# In[11]:


def writeToExcel(targetLabels, predictedLabels):
    df = pd.DataFrame({'Target':targetLabels,'Predicted': predictedLabels})
    df.to_excel('./Data/AI-Design-RFModel.xlsx', sheet_name='RFModel', index=False)
    return


# In[12]:


def drawErrorHistogram(predictions, targets):
    #printing the error
    error = predictions - targets
    plt.grid(b=False)
    plt.hist(error, bins = 25, rwidth=0.8,color="red", range=[-0.2, 0.2], align='mid')
    plt.xlabel("Prediction Error (MSE)", fontdict=font)
    plt.ylabel("Occurrences", fontdict=font)
    return


# In[13]:


def saveRFModel(rfModel):
    with open('./RFModel', 'wb') as RFModelfile:
        cPickle.dump(rfModel, RFModelfile)
    print("Saved model to disk")
    return


# In[14]:


#load saved RFmodel

loadedRFModel = loadRFmodel("RFModel")
#use the predict function to predict the targetLabels
loadedRFModel, mse, predictions = predictUsing(loadedRFModel, testFeatures, targetTestLabels.ravel())

#flatten the predictions array
predictions = predictions.flatten()
targetTestLabels = targetTestLabels.flatten()

#plot the error histogram

drawErrorHistogram(predictions, targetTestLabels)

print("MSE: ", mse)

print("First Predicted Label : ", predictions[0], "First Test Label: ",targetTestLabels[0])

print('R2_Score:', r2_score(targetTestLabels, predictions))


# In[15]:


#BuildModel Function Definition

def buildRFBaseModelUsing(trainingFeatures, targetLabels, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Instantiate model with 100 decision trees
    rf = RandomForestRegressor(n_estimators = n_estimators, criterion = 'mse', max_depth = max_depth, min_samples_split = min_samples_split, bootstrap = True, random_state = 42)
    # Train the model on training data
    rf.fit(trainingFeatures, targetLabels.ravel());
    return rf


# In[56]:


def estimatorsMSE():
    n_estimators = [10,20,30,40,50,60,70,80,90,100]
    trainingMSEResults = []
    testMSEResults = []
    for estimator in n_estimators:
        rfModel = buildRFBaseModelUsing(trainingFeatures, targetLabels, estimator,None,2,1)
        #predict using the trained model on training features and targetLabels
        rfModel, testMSE, predictions = predictUsing(rfModel, testFeatures, targetTestLabels.ravel())
        #predict using the traingin set y values
        rfModel, trainMSE, predictions = predictUsing(rfModel, trainingFeatures, targetLabels.ravel())
        #append the training mse results into the results array
        trainingMSEResults.append(trainMSE)
        #append the training mae results into the results array
        testMSEResults.append(testMSE)
    return n_estimators, trainingMSEResults, testMSEResults


# In[57]:


#Plot estimators vs MSE

n_estimators, trainingMSEResults, testMSEResults = estimatorsMSE()

print("TestMSE:", testMSEResults)

#save MSE results to text file
np.savetxt("NumberofEstimators-MSEresults.txt", testMSEResults, delimiter=",")

line1, = plt.plot(n_estimators, testMSEResults, "r", label = "MSE")
#line2, = plt.plot(n_estimators, testMSEResults, "b", label = "Train MSE")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, frameon=False)
plt.ylabel("Mean Squared Error (MSE)", fontdict=font)
plt.xlabel("Number of estimators", fontdict=font)
plt.show()


# In[60]:


def maxDepthMSE():
    max_Depth = [2,4,6,8,10,12,14,16,18,20,22,24]
    trainingMSEResults = []
    testMSEResults = []
    for depth in max_Depth:
        rfModel = buildRFBaseModelUsing(trainingFeatures, targetLabels, 70, depth, 2, 1)
        #predict using the trained model on training features and targetLabels
        rfModel, testMSE, predictions = predictUsing(rfModel, testFeatures, targetTestLabels.ravel())
       #predict using the traingin set y values
        rfModel, trainMSE, predictions = predictUsing(rfModel, trainingFeatures, targetLabels.ravel())
        #append the training mse results into the results array
        trainingMSEResults.append(trainMSE)
        #append the training mae results into the results array
        testMSEResults.append(testMSE)
    return max_Depth, trainingMSEResults, testMSEResults


# In[61]:


#Plot estimators vs MSE

max_Depth, trainingMSEResults, testMSEResults = maxDepthMSE()

print("TestMSE:", testMSEResults)

#save MSE results to text file
np.savetxt("MaxDepth-MSEresults.txt", testMSEResults, delimiter=",")

line1, = plt.plot(max_Depth, testMSEResults, "r", label = "MSE")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, frameon=False)
plt.ylabel("Mean Squared Error(MSE)", fontdict=font)
plt.xlabel("Max Depth", fontdict=font)
plt.show()


# In[62]:


def minSamplesSplitMSE():
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    trainingMSEResults = []
    testMSEResults = []
    
    for sample_split in min_samples_splits:
        rfModel = buildRFBaseModelUsing(trainingFeatures, targetLabels, 20, 10, sample_split, 1.0)
        #predict using the trained model on training features and targetLabels
        rfModel, testMSE, predictions = predictUsing(rfModel, testFeatures, targetTestLabels.ravel())
       #predict using the traingin set y values
        rfModel, trainMSE, predictions = predictUsing(rfModel, trainingFeatures, targetLabels.ravel())
        #append the training mse results into the results array
        trainingMSEResults.append(trainMSE)
        #append the training mae results into the results array
        testMSEResults.append(testMSE)
    return min_samples_splits, trainingMSEResults, testMSEResults


# In[63]:


#Plot min_samples_split vs MSE

min_samples, trainingMSEResults, testMSEResults = minSamplesSplitMSE()

print("MSE: ", testMSEResults)

#save MSE results to text file
np.savetxt("MinSamplesSplit-MSEresults.txt", testMSEResults, delimiter=",")

line1, = plt.plot(min_samples, testMSEResults, "r", label = "MSE")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, frameon=False)
plt.ylabel("Mean Squared Error (MSE)", fontdict=font)
plt.xlabel("Minimum Samples Split", fontdict=font)
plt.show()


# In[64]:


def minSamplesLeaveMSE():
    min_samples_leaf = np.linspace(0.1, 0.8, 8, endpoint=True)
    trainingMSEResults = []
    testMSEResults = []
    for samples_leave in min_samples_leaf:
        rfModel = buildRFBaseModelUsing(trainingFeatures, targetLabels, 20, 8, 2, min_samples_leaf)
        #predict using the trained model on training features and targetLabels
        rfModel, testMSE, predictions = predictUsing(rfModel, testFeatures, targetTestLabels.ravel())
       #predict using the traingin set y values
        rfModel, trainMSE, predictions = predictUsing(rfModel, trainingFeatures, targetLabels.ravel())
        #append the training mse results into the results array
        trainingMSEResults.append(trainMSE)
        #append the training mae results into the results array
        testMSEResults.append(testMSE)
    return min_samples_leaf,trainingMSEResults, testMSEResults


# In[65]:


#Plot min_samples_split vs MSE

min_samples_leaf,  trainingMSEResults, testMSEResults = minSamplesLeaveMSE()

print("MSE: ", testMSEResults)

#save MSE results to text file
np.savetxt("MinSamplesLeaf-MSEresults.txt", testMSEResults, delimiter=",")
    
line1, = plt.plot(min_samples_leaf, testMSEResults, "r", label = "MSE")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)},frameon=False)
plt.ylabel(" Mean Squared Error (MSE)", fontdict=font)
plt.xlabel("Minimum Samples Leaf", fontdict=font)
plt.show()


# In[17]:


def buildDefalutRFModelUsing(trainingFeatures, targetLabels):
    # Instantiate model with 100 decision trees
    rf = RandomForestRegressor(bootstrap = True, random_state = 1)
    # Train the model on training data
    rf.fit(trainingFeatures, targetLabels);
    return rf


# In[18]:


# buildDefaultModel
rfModel = buildDefalutRFModelUsing(trainingFeatures, targetLabels)

#use the predict function to predict the targetLabels
rfModel, testingMSEResult, predictions = predictUsing(rfModel, testFeatures, targetTestLabels.ravel())

#flatten the predictions array

predictions = predictions.flatten()
targetTestLabels = targetTestLabels.flatten()

#Write the predicted values to excel sheet

writeToExcel(targetTestLabels, predictions)

#plot the error histogram

drawErrorHistogram(predictions, targetTestLabels)

print("MSE: ", testingMSEResult)

print("First Predicted Label : ", predictions[0], "First Test Label: ",targetTestLabels[0])

print('R2_Score:', r2_score(targetTestLabels, predictions))


# In[33]:


# buildModel
rfModel = buildRFBaseModelUsing(trainingFeatures, targetLabels, 20, 8, 2, 1.0)

#use the predict function to predict the targetLabels
rfModel, testingMSEResults, predictions = predictUsing(rfModel, testFeatures, targetTestLabels.ravel())

#flatten the predictions array
predictions = predictions.flatten()
targetTestLabels = targetTestLabels.flatten()

#Write the predicted values to excel sheet

writeToExcel(targetTestLabels, predictions)

#plot the error histogram

drawErrorHistogram(predictions, targetTestLabels)

#save the model to disk

saveRFModel(rfModel)

print("MSE: ", testingMSEResults)

print("First Predicted Label : ", predictions[0], "First Test Label: ",targetTestLabels[0])

print('R2_Score:', r2_score(targetTestLabels, predictions))


# In[34]:


#Visualise the important parameters
showImportantParametersUsing(rfModel, trainingFeaturesNames)


# In[ ]:




