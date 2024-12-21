# Analysis-of-EMG-data-from-MYO-armband.-
This project investigates machine learning approaches for classifying four hand gestures—rock (0), scissors (1), paper (2), and ok (3)—using EMG data from the MYO armband. Several algorithms were explored, including Naïve Bayes,Support Vector Machine (SVM), and K-Nearest Neighbors (KNN), to evaluate their performance in gesture recognition. The study aims to compare these models' classification accuracy and effectiveness, providing insights into their suitability forEMG-based gesture recognition tasks.



# import all libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# import datasets

df0 = pd.read_csv("0.csv", header=None )
df1 = pd.read_csv("1.csv", header=None )
df2 = pd.read_csv("2.csv", header=None )
df3 = pd.read_csv("3.csv", header=None )
df = pd.concat([df0,df1,df2,df3], axis = 0)
df.head()

# Find missing value
df.isnull().sum()
# shape of dataset
print('The Shape of our dataset is: ',df.shape)
df.dtypes

x = df.loc[:,0:63]
y = df[64]

# train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# preprocessing 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train))
x_test = pd.DataFrame(sc.transform(x_test))

# models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

models=[SVC(),GaussianNB(),KNeighborsClassifier()]

model_names=["Support Vector Machine","Gaussian Naive Bayes","K-Nearest Neighbors"]
models_scores=[]
for model,model_name in zip(models,model_names):
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
models_scores.append([model_name,accuracy])
sorted_models_scores=sorted(models_scores,key=lambda x:x[1],reverse=True)
for model in sorted_models_scores:
print("Accuracy Score: ",f'{model[0]}: {model[1]*100:.2f}')

# classification 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print('Classification Report: \n', classification_report(y_test,y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test,y_pred))
