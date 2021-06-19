#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.set_printoptions(suppress=True)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.25
import tensorflow as tf
from openpyxl import load_workbook


font = {'family': 'Arial',
        'weight': 'normal',
        'size': 12,
        }

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV


# In[2]:


# LoadData set function definition

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

    # accessing the values of the trainingSetColumns
    trainingFeaturesDF = dataCopy.iloc[:,0:4]
    
    # changing trainingDF to Numpy array
    trainingFeatures = trainingFeaturesDF.values
    
     # accessing the targetColumns
    targetLabelsColumn = dataCopy.columns[-1]
    
     # accessing the targetColumns
    targetLabelsDF = dataCopy.loc[:,targetLabelsColumn]
    
    # changing targetDF to Numpy array
    targetLabels = targetLabelsDF.values
    
    #return the values 
    return (trainingFeatures, targetLabels, dataCopy)


# In[3]:


#Load the data
trainingFeatures, targetLabels, datacopy = loadData('./Data','/RSM-Data.xlsx')

#printing the dimensions of the data
print("Dimensions: TrainingFeatures Data shape is:", trainingFeatures.shape)
print("Dimensions: TargetLabels Data shape is:", targetLabels.shape)

# plotting correlation matrix for all the variables
#fig, ax = plt.subplots(figsize=(15,10))
#sns.heatmap(datacopy.corr(), annot=True, linewidth = 0.5, cmap='Blues', ax=ax)

#Pair Plot
#sns.pairplot(datacopy,height=3)


# In[4]:


def normaliseData (trainingFeatures, targetLabels):
    """
    Function to scale the given X, Y
    X: Input data, scaled using the sklearn preprocessing
    Y: Ouptu data, scaled using the sklearn preprocessing
    """
    # access the scalar for input
    scalerX = MinMaxScaler()
    # fit the data to the scaler to scale the input data set 
    scalerX = scalerX.fit(trainingFeatures)
    # transform it to make the input normalised
    trainingFeaturesScaled = scalerX.transform(trainingFeatures)
    
    # reshaping the targetY as it has only one feature or label
    targetLabels = targetLabels.reshape(-1, 1)
    
    # access the scalar for output
    scalerY = MinMaxScaler()
    # fit the data to the scaler to scale the output data set 
    scalerY = scalerY.fit(targetLabels)
    # transform it to make the output normalised
    targetLabelsScaled = scalerY.transform(targetLabels)
    
    #print(trainXScale[100:])
    #print(targetYScale[:100])
    
    return trainingFeaturesScaled, targetLabelsScaled


# In[5]:


# scale data
trainingFeaturesScaled, targetLabelsScaled = normaliseData(trainingFeatures, targetLabels)

#printing the dimensions of the data
print("Dimensions: TrainingFeatures Scaled Data shape is:", trainingFeaturesScaled.shape)
print("Dimensions: TargetLabels Scaled Data shape is:", targetLabelsScaled.shape)


# In[6]:


def splitData (trainingFeaturesScaled, targetLabelsScaled):
    """
    Split the data using the sklearn preprocessing tool
    Splitting will be done two times, to get training data, evaluation data, test data
    trainXScale: the parameter where the scaled input is fed to be split as 70, 15, 15 (train, evaluate, test)
    targetYScale: The parameter where the scaled output is fed to be split as 70, 15, 15
    
    """
    
    # use the sklearn split function to split, the sklearn gives only 
    trainingFeatures, trainingFeaturesValAndTest, targetLabels, targetLabelsScaledValAndTest = train_test_split(trainingFeaturesScaled, targetLabelsScaled, test_size=0.3, random_state=42)
    
    # We are again splitting the data to bring our evaluation set
    trainingFeaturesEval, testFeatures, targetLabelsEval, testLabels = train_test_split(trainingFeaturesValAndTest, targetLabelsScaledValAndTest, test_size=0.5, random_state=42)
    
    # printing the shape of the vectors
    #print(trainX.shape, trainXEval.shape, trainXTest.shape, targetY.shape, targetYEval.shape, targetYTest.shape)
    
    return trainingFeatures, trainingFeaturesEval, testFeatures, targetLabels, targetLabelsEval, testLabels


# In[7]:


# split data
trainingFeatures, trainingFeaturesEval, testFeatures, targetLabels, targetLabelsEval, testLabels = splitData(trainingFeaturesScaled, targetLabelsScaled)

# printing the shapre of the data for confirmation

print("Dimensions: Training Dataset Shape:", trainingFeatures.shape)
print("Dimensions: Training Target Dataset Shape:", targetLabels.shape)

print("Dimensions: Evaluation Training Features Dataset Shape:", trainingFeaturesEval.shape)
print("Dimensions: Evaluation Target Dataset Shape:", targetLabelsEval.shape)

print("Dimensions: Test Features Dataset Shape:", testFeatures.shape)
print("Dimensions: Test Target Dataset Shape:", testLabels.shape)


# In[8]:


def loadMLPmodel(mlpModelName):
    loaded_model = load_model(mlpModelName)
    return loaded_model


# In[9]:


#Prediction Function Definition
def predictUsing(trainedNNModel, features, labels):
    # predict the values for the given input using trainedNNModel
    predictions = trainedNNModel.predict(features)
    #calculate the mean_square_error
    loss = mean_squared_error(predictions, labels)
    print("MSE: ", loss)
    # compute the difference between the *predicted* CC and the *actual* CC
    errors = abs(predictions - labels)
    return predictions


# In[10]:


def evaluateTestLossUsing(testFeatures, testLabels, trainedNNModel):
    testloss = trainedNNModel.evaluate(testFeatures, testLabels, verbose=2)
    return testloss


# In[11]:


def writeToExcel(targetLabels, predictedLabels):
    df = pd.DataFrame({'Target':targetLabels,'Predicted': predictedLabels})
    df.to_excel('./Data/AI-Design-MLPModel.xlsx', sheet_name='MLPModel', index=False)
    return


# In[12]:


def drawTrainingVsEvaluationLossUsing(hist):
    plt.plot(hist.history['loss'], color='r')
    plt.plot(hist.history['val_loss'], color ='k')
    plt.ylabel('Loss', fontdict=font)
    plt.xlabel('Epoch', fontdict=font)
    plt.legend(['Training Loss', 'Evaluation Loss'], loc='upper right', frameon=False)
    plt.show()
    trainingLossHistory = np.array(hist.history['loss'])
    validationLossHistory = np.array(hist.history['val_loss'])
    np.savetxt("trainingLossHistory.txt", trainingLossHistory, delimiter=",")
    np.savetxt("validationLossHistory.txt", validationLossHistory, delimiter=",")
    return


# In[13]:


def drawErrorHistogram(predictions, targets):
    #printing the error
    error = predictions - targets
    plt.ylim((0,250))
    plt.hist(error, bins = 25, rwidth=0.8,color="r", range=[-0.2, 0.2], align='mid')
    plt.xlabel("Prediction Error", fontdict=font)
    plt.ylabel("Occurrences", fontdict=font)
    return


# In[14]:


def saveModel(nnModel,name):
    nnModel.save(name)
    print("Saved model to disk")
    return


# In[15]:


#load saved model
savedModel = loadMLPmodel("MLP-Model.h5")

#print savedModel summary
print(savedModel.summary())

predictions = predictUsing(savedModel, testFeatures, testLabels)

#Flattern the prediction and testLabels

predictions = predictions.flatten()
testLabels = testLabels.flatten()

#draw error histogram

drawErrorHistogram(predictions, testLabels)
#Write it to the excel sheet

writeToExcel(testLabels, predictions)


# In[20]:


savedModel.weights[0]


# In[16]:


def buildMLPModelUsing(inputShape=4, neuronsInputLayer=10, neuronsHiddenLayer=5, numberofLayers=1):
    """
    Building a basic model with Linear->Relu as the activation function for the hidden layers
    The final layer is made with Linear only to find the continous value of the output
    
    """
    # getting access to a model object and intialising the number of layers with appropriate activation function
    nnModel = Sequential()
    nnModel.add(tf.keras.layers.InputLayer(input_shape = [inputShape]))
    nnModel.add(Dense(neuronsInputLayer, activation='relu'))
    for layers in range(numberofLayers):
        nnModel.add(Dense(neuronsHiddenLayer, activation='relu'))
    
    nnModel.add(Dense(1))
        
    # compile the model with the best optimiser and loss function to build our model
    
    nnModel.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=["mean_squared_error"])
    
    # return the untrainedModel
    return nnModel


# In[17]:


def fitMLPModelUsing(nnModel, epoch, trainingFeatures, targetLabels, trainingFeaturesEval, targetLabelsEval, modelcallBacks):
    hist = nnModel.fit(trainingFeatures, targetLabels, epochs=epoch,
                       validation_data=(trainingFeaturesEval, targetLabelsEval))
    #hist = nnModel.fit(trainingFeatures, targetLabels, epochs=epoch,callbacks= modelcallBacks,
            #validation_data=(trainingFeaturesEval, targetLabelsEval))
    return hist


# In[18]:


# Set callback functions to early stop training and save the best model so far
modelcallBacks = [EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=10),  ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)]


# In[21]:


# build the model
trainedNNModel = buildMLPModelUsing(4,10,5,1)
#fit the model
trainedNNModelHist = fitMLPModelUsing(trainedNNModel, 500, trainingFeatures, targetLabels, trainingFeaturesEval, targetLabelsEval, modelcallBacks)


# In[24]:


#draw TrainingVsEvaluationLoss
drawTrainingVsEvaluationLossUsing(trainedNNModelHist)


# In[83]:


#evaluate model against test loss
testloss = evaluateTestLossUsing(testFeatures, testLabels, trainedNNModel)


# In[84]:


#predict testLabels using trainedNNModel

predictions = predictUsing(trainedNNModel, testFeatures, testLabels)

#Flattern the prediction and testLabels

predictions = predictions.flatten()
testLabels = testLabels.flatten()

#draw error histogram

drawErrorHistogram(predictions, testLabels)

#Write it to the excel sheet

writeToExcel(testLabels, predictions)

print(r2_score(testLabels, predictions))


# In[85]:


#savemodel
saveModel(trainedNNModel,"MLP-Model.h5")


# In[ ]:




