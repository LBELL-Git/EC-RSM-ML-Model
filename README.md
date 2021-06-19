# Modelling electrical conduction in resistive-switching-memory through machine learning

# Description of the project

The motivation of the project is to model the electrical conduction of resistive switching memories (RSM) using machine learning (ML) algorithms. Here, we used multi-layer perceptron, random forest machine learning algorithms to model the electrical conduction of aluminum nitride (AlN) RSM devices.

The obtained data to model the electrical conduction of AlN resistive switching memories are measured using the standard measurement instruments. The processed and collected date amounts to approximately 3100 data points and can be found in every respective model’s Data folder. Further details on measurement and data collections procedures are available online.

Standard machine learning python libraries and Jupiter notebooks are adopted to train the ML model. The trained models are in SourceCode/ folder of every RSM device. In addition, the SourceCode/ folder consists of python file to train a new model or to load a trained model to do inference on them using the test data. 

While we execute the Jupiter notebooks, the results and figures are displayed in the notebook console. Figures that are in the Figures/ folder are part of the accepted manuscript and supplementary information of the project. 

# Features to be implemented in the future

In future, the data set could be expanded to accommodate the modelling of various RSM devices. Currently, we use the trained models to perform inference on other RSM devices such as TiO2 and SiOx devices. However, the amount of data is not enough to develop a complete model. To address this, a wide variety of data set from various RSM devices could improve the existing model. Furthermore, object-oriented programming methodologies could be adopted to make the available base code a reusable code.

# Installing and using the project

The project is hosted in a public repository and can be cloned using the GitHub link. 

1.	Load the source code python file from SourceCode/ folder of either MLP model or RF Model to a Jupiter notebook and execute the functions in a sequential manner.


