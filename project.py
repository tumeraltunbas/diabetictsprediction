#library imports
import pandas as pd #for data processing
import numpy as np
from sklearn.model_selection import train_test_split # for split to df
from sklearn.linear_model import LogisticRegression #for use logistic regression
from sklearn.metrics import confusion_matrix
#Data understanding
df = pd.read_csv('C:\Kamp\Python\Python-Project\diabetes2.csv') #Reading file
temp = df.head() #Bring first 5 record
temp = df.shape #768 rows and 9 column (768,9) 
temp = df.isnull().sum() #There are no, not a number value in our dataset.
temp = df.dtypes #It shows datatypes of columns
temp = df.describe() #Some min values starts from 0, that's mean, there are lack of data.
temp = (df==0).sum() #how many 0 values ​​are in each column
'''
Pregnancies                 111
Glucose                       5
BloodPressure                35
SkinThickness               227 
Insulin                     374
BMI #Body mass index         11
DiabetesPedigreeFunction      0
Age                           0
Outcome                     500
dtype: int64
'''
#Glucose, SkinThickness BMI,Insulin,BloodPressure and age should be greater than 0
#That's why we gonna replace 0 values as a NaN 
columns = ['BMI','Age','Insulin','Glucose','BloodPressure','SkinThickness' ]
for cl in columns: #each cl in columns.
    df[cl] = df[cl].replace(0,np.nan) #we replaced 0 values to NaN
df=df.dropna() #Dropping rows which contains NaN value.
#Analyze
x=df.drop("Outcome",axis=1) #Independent varies in x
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100) 
#test data includes average 0.25 of all data but train data includes 0.75 of all data
lg_model=LogisticRegression(max_iter=1000)
lg_model.fit(x_train,y_train)
temp = lg_model.score(x_test,y_test)
#The accuracy score of test modal = 0.7857142857142857
temp = lg_model.score(x_train,y_train)
#The accuracy score of train modal = 0.782312925170068
guess = lg_model.predict(x_test)
#accuracy rate of the model
temp=confusion_matrix(y_test,guess)
print(temp)
'''
[60  8]
[13 17]
'''
