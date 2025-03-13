#DIABETES ANALYSER
#diabetes data logistic regression

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 

file=r"C:\Users\ABCD\Downloads\diabetes\diabetes.csv"
data=pd.read_csv(file)

data.head()

X=data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y=data["Outcome"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))  
print(classification_report(y_test, predictions))  

pregnancies=int(input("Enter the pregnancies: "))
glucose=int(input("Enter the glucose level: "))
Blood_pressure=int(input("Enter the blood pressure level: "))
skin_thickness=int(input("Enter the skin thickness: "))
insulin=int(input("Enter the insulin level: "))
bmi=float(input("Enter the bmi: "))
diabetes_pred_function=float(input("enter diabetes pedigree function: "))
age=int(input("Enetr the age:"))

value=pd.DataFrame([[pregnancies,glucose,Blood_pressure,skin_thickness,insulin,bmi,diabetes_pred_function,age]],
                       columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])

new_prediction=model.predict(value)
print("Diabetes prediction: ","Yes,You have diabetes take prescription and medicines" if new_prediction[0]==1  else "No You dont have diabetes!!!")

#THIS PROJECT IS A DIABETES ANALYZER .IT'S BASICALLY BASED ON LOGISTIC REGRESSION WHICH IS A VERY USEFUL TECHNIQUE IN BUILDING MACHINE LEARNING MODELS.
#HOPEFULLY YOU FIND IT INTERESTING!!!!!!