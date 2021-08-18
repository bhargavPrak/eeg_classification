# eeg_classification
The supplementary material for the publication: A Novel Approach to Learning Models on EEG data Using
Graph Theory features - A Comparative Study (doi:)

 This is the code dump used in studying different datasets with the help of 4 different classification models:
 - Logistic Regression:
   Logistic Regression model with Gaussian Kernel and Laplacian Prior. The Gaussian kernel optimizes the separation between data points in the transformed space obtained in preprocessing, while the Laplacian Prior enhances the sparseness of learned L.R. regressors to avoid overfitting. 
 - Random Forest Classifier:
   Creation of many random decision trees, each predicting a particular class according to241the features given to it. Once each tree predicts a class, voting is carried out to take into consideration the final class according to a majority.
 - Support Vector Machine:
   Finding a hyperplane that separates the different classes with the largest margin while keeping the misclassification as low as possible, by minimizing the cost/objective function.
 - Recurrent Neural network:
   The first layer is an LSTM layer with 256 memory units, and it defines the input shape. The next layer is a Dense layer with a sigmoid activation function. A dropout layer is applied after each LSTM layer to avoid over-fitting of the model. Validation accuracy is done using loss as binary cross-entropy, optimizer as adam and metrics as accuracy.
   
   NOTE: The models were run multiple times using different hyper-parameters to get the best possible accuracies. The hyper-parameters were tuned using standard techinques such as Random Search and Grid Search, hit and trial methods, brute force approach. This may needed to be done to reproduce the results.
   
 Datasets used in the study:
 - DASS21 Data (in-house): https://drive.google.com/drive/folders/1jjHGBi1j3puIaMCTyWRFflZYElaRGaNq?usp=sharing
 - Visual Working memory:
 - Verbal Working memory:
 - Selection and Depression:
 Since different datasets were used during the course of this study, this code only represents the definition of the models used, and need to be modified as per the dataset properties. 
 
