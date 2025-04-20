# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("C:\\Users\\balak\\Downloads\\Placement_Data.csv")
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
![image](https://github.com/user-attachments/assets/5ab6f643-c1e4-4e2e-b5a5-ac8a7cac0b11)

![image](https://github.com/user-attachments/assets/be6e82a2-8bcd-4126-837c-d24a8c761256)

![image](https://github.com/user-attachments/assets/921bcb37-52b4-4f36-9f21-689556ace082)

![image](https://github.com/user-attachments/assets/ff01781b-7033-4379-8ed7-4d95b03d69f3)

![image](https://github.com/user-attachments/assets/dd6a59f1-468d-407f-ac32-cdbd172e535c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
