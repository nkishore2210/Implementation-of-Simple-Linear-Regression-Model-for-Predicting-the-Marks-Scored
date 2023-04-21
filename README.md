# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3.  Implement training set and test set of the dataframe
4.  Plot the required graph both for test data and training data.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: N KISHORE
RegisterNumber:  212222240049

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
print(x)

y=df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
![image](https://user-images.githubusercontent.com/118707090/233581527-4c4191ea-b507-499e-8be3-9adc07cf373d.png)
![image](https://user-images.githubusercontent.com/118707090/233581586-c9e9af43-7b05-4b21-b91c-2d4276054293.png)
![image](https://user-images.githubusercontent.com/118707090/233581617-7dc11702-c93c-4b39-a201-ec3ac437b7cc.png)
![image](https://user-images.githubusercontent.com/118707090/233581657-e9b37243-1d7d-480f-9814-e24fa843b08f.png)
![image](https://user-images.githubusercontent.com/118707090/233581767-fe4b878a-3afa-407b-ab8b-b197b0f54622.png)
![image](https://user-images.githubusercontent.com/118707090/233581796-b781b978-4d0b-4fcd-a6b9-fdde253dce2c.png)
![image](https://user-images.githubusercontent.com/118707090/233581864-529eadf6-6672-435f-b6a2-c9093daa9fce.png)
![image](https://user-images.githubusercontent.com/118707090/233581895-eae1173d-aff3-4456-a12b-1b3650bca688.png)
![image](https://user-images.githubusercontent.com/118707090/233581919-00148f75-91c1-4cfb-ba8d-3d64415ae6fe.png)
![image](https://user-images.githubusercontent.com/118707090/233581953-f242001f-c008-49bd-89ba-ee0cbee80ad8.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
