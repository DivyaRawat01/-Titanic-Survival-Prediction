import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def predictSurvivalRate(model,indata):
    predict = model.predict(indata)
    print(f"\n--------------------------------")
    print(f"Your Survival Prediction: {predict[0]}")
    print("1 = Survived, 0 = Did Not Survive")
    print("--------------------------------")

model = joblib.load("model.pkl")

id = int(input("enter PassengerID:"))
name = input("enter your name:")
sex = input("enter your gender (M/F):")
pclass = int(input("enter Passenger class(1/2/3):"))
age = int(input("enter your age:"))
sibSp = int(input("enter no of sibiling borded:"))
parch = int(input("enter no of parent borded:"))
fare = float(input("Enter Fare amount you pay:"))
embarked = input("Enter embarked:")

fare_log = np.log1p(fare)
if sex[0].lower() == 'f':
    sex = 1
else: 
    sex = 0

familysize = sibSp + parch + 1
isalone = 0
if familysize == 1:
    isalone = 1

embarked_Q = 1 if embarked.upper() == "Q" else 0
embarked_S = 1 if embarked.upper()== "S" else 0
   
pclass_2 = 1 if pclass == 2 else 0
pclass_3 = 1 if pclass == 3 else 0

inputData = pd.DataFrame({
    'sex': [sex],
    'age': [age],
    'embarked_Q': [embarked_Q],
    'embarked_S': [embarked_S],
    'pclass_2': [pclass_2],
    'pclass_3': [pclass_3],
    'familysize': [familysize],
    'isalone': [isalone],
    'fare_log': [fare_log]
})

predictSurvivalRate(model,inputData)



