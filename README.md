# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries & load data using pandas, and preview with df.head().

2.Clean data by dropping sl_no and salary, checking for nulls and duplicates.

3.Encode categorical columns (like gender, education streams) using LabelEncoder.

4.Split features and target:

X = all columns except status

y = status (Placed/Not Placed)

5.Train-test split (80/20) and initialize LogisticRegression.

6.Fit the model and make predictions.

7.Evaluate model with accuracy, confusion matrix, and classification report. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M ALMAAS JAHAAN
RegisterNumber:  212224230016
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data.csv")
data.head()

# Copy data and drop unnecessary columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

# Check for missing values and duplicates
data1.isnull().sum()
data1.duplicated().sum()

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

data1.head()

# Separate features and target
x = data1.iloc[:, :-1]  # Features
y = data1["status"]     # Target (Placement Status)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize and train Logistic Regression model
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

# Predict on test data
y_pred = lr.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)

# Predict placement status for a new student
new_data = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
placement_status = lr.predict(new_data)
print("Predicted Placement Status:", placement_status)
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![image](https://github.com/user-attachments/assets/73a69e47-1712-4a81-99e5-cbb23b84b69f)

![image](https://github.com/user-attachments/assets/3c4de668-dafb-40f7-a525-a6108816aba2)

![image](https://github.com/user-attachments/assets/28369ea3-2286-4ea3-9f45-feadceba4d62)

![image](https://github.com/user-attachments/assets/d52f3595-ccac-4646-aa43-a9c70de65f4f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
