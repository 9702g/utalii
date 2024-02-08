
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn import metrics
data=pd.read_csv("V:\\Git_repo\\utalii\\Kenyan_data.csv")
# Clearing errors and missing values from data set
data['travel_with'] = data['travel_with'].fillna('Alone')
data['total_female'] = data['total_female'].fillna(1.0)
data['total_male'] = data['total_male'].fillna(1.0)
data['most_impressing'] = data['most_impressing'].fillna('No comments')
data['age_group'] = data['age_group'].replace('24-Jan', '1-24')
data["total_female"] = data['total_female'].astype('int')
data["total_male"] = data['total_male'].astype('int')
data["nights_stayed"] = data['nights_stayed'].astype('int')
data["total_people"] = data["total_female"] + data["total_male"]
data.isnull().sum()

data["total_female"] = data['total_female'].astype('int')
data["total_male"] = data['total_male'].astype('int')
data["nights_stayed"] = data['nights_stayed'].astype('int')
data["total_people"] = data['total_people'].astype('int')
data.country = data.country.replace(["SWIZERLAND", "BURGARIA", "MALT"], ["SWITZERLAND", "BULGARIA", "MALTA"])
data.country = data.country.replace(["DRC", "SCOTLAND", "UAE", "PHILIPINES", "DJIBOUT", "MORROCO"],
                                    ["DEMOCRATIC REPUBLIC OF THE CONGO", "UNITED KINGDOM", "UNITED ARAB EMIRATES",
                                     "PHILIPPINES", "DJIBOUTI", "MOROCCO"])
data.info_source = data.info_source.replace("Tanzania Mission Abroad", "Kenya Tourist Board")
data.age_group = data.age_group.replace("Jan-24", "1-24")

data.head(20)

data.isnull().sum()
# %then it's time to encode objects into numeric

for colname in data.select_dtypes("object"):
    data[colname],_=data[colname].factorize()
# Now all columns that can be converted to numeric have been converted
# Step 3 model building
import warnings
warnings.filterwarnings('ignore')
x=data.drop(['total_cost'], axis=1)
y=data['total_cost']
#splitting the data into training data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2,random_state= 2)
#model Training
#XGBOOST REGRESSOR
#loading the model
model=XGBRegressor()
#training the model with x_train
# Instantiate an object of XGBModel class
model = XGBRegressor()
# Call the fit method on the instantiated object
model.fit(X=x_train, y=y_train)
#Evaluation 
training_data_prediction=model.predict(x_train)
#R squared error 
score_1=metrics.r2_score(y_train,training_data_prediction)
#find the variants btwn both 
#mean absolute error 
score_2= metrics.mean_absolute_error(y_train,training_data_prediction)
#find difference and give mean
print('R squared error :', score_1)
print('Mean absolute Error:', score_2)
#prediction on training data
test_data_prediction=model.predict(x_test)
#R squared error 
score_1=metrics.r2_score(y_test,test_data_prediction)
#find the variants btwn both 
#mean absolute error 
score_2= metrics.mean_absolute_error(y_test,test_data_prediction)
#find difference and give mean
print('R squared error :', score_1)
print('Mean absolute Error:', score_2)
#visualizing the actual prices and predicted prices
#y train and y test
# Save the trained model to a .pkl file using pickle


import pickle

Save the trained model to a .pkl file using pickle
model_file_path = ("V:\\my project\\xgb_model.pkl")
with open(model_file_path, 'wb') as file:
    pickle.dump(model, file)

print("Model saved as", model_file_path)


import streamlit as st
import pandas as pd
import os
import pickle

# Header
st.header("Kenya Tourism Expenditure Prediction")
st.subheader("A simple machine learning app to predict how much money a tourist will spend when visiting Kenya.")
st.image("V:\\Git_repo\\utalii\\images (1).jpg")

# Form
my_form = st.form(key="financial_form")

# Function to transform Yes and No options
@st.cache
def func(value):
    return "Yes" if value == 1 else "No"

import streamlit as st

# Input for country
country = st.selectbox("Country", [
    "SWITZERLAND", "UNITED KINGDOM", "CHINA", "SOUTH AFRICA", "UNITED STATES OF AMERICA",
    "NIGERIA", "INDIA", "BRAZIL", "CANADA", "MALTA", "MOZAMBIQUE", "RWANDA", "AUSTRIA",
    "MYANMAR", "GERMANY", "KENYA", "ALGERIA", "IRELAND", "DENMARK", "SPAIN", "FRANCE",
    "ITALY", "EGYPT", "QATAR", "MALAWI", "JAPAN", "SWEDEN", "NETHERLANDS", "UAE", "UGANDA",
    "AUSTRALIA", "YEMEN", "NEW ZEALAND", "BELGIUM", "NORWAY", "ZIMBABWE", "ZAMBIA", "CONGO",
    "BULGARIA", "PAKISTAN", "GREECE", "MAURITIUS", "DRC", "OMAN", "PORTUGAL", "KOREA",
    "SWAZILAND", "TUNISIA", "KUWAIT", "DOMINICA", "ISRAEL", "FINLAND", "CZECH REPUBLIC",
    "UKRAINE", "ETHIOPIA", "BURUNDI", "SCOTLAND", "RUSSIA", "GHANA", "NIGER", "MALAYSIA",
    "COLOMBIA", "LUXEMBOURG", "NEPAL", "POLAND", "SINGAPORE", "LITHUANIA", "HUNGARY",
    "INDONESIA", "TURKEY", "TRINIDAD AND TOBAGO", "IRAQ", "SLOVENIA", "UNITED ARAB EMIRATES",
    "COMORO", "SRI LANKA", "IRAN", "MONTENEGRO", "ANGOLA", "LEBANON", "SLOVAKIA", "ROMANIA",
    "MEXICO", "LATVIA", "CROATIA", "CAPE VERDE", "SUDAN", "COSTA RICA", "CHILE", "NAMIBIA",
    "TAIWAN", "SERBIA", "LESOTHO", "GEORGIA", "PHILIPPINES", "IVORY COAST", "MADAGASCAR",
    "DJIBOUTI", "CYPRUS", "ARGENTINA", "URUGUAY", "MOROCCO", "THAILAND", "BERMUDA", "ESTONIA",
    "BOTSWANA", "VIETNAM", "GUINEA", "MACEDONIA", "HAITI", "LIBERIA", "SAUDI ARABIA", "BOSNIA",
     "PERU", "BANGLADESH", "JAMAICA", "SOMALIA"
])

# Input for age group
age_group = st.selectbox("Age Group", ["1-24", "25-44", "45-64", "65+"])

# Input for travel with
travel_with = st.selectbox("Travel With", ["Friends/Relatives", "Alone", "Spouse", "Children", "Spouse and Children"])

# Input for total number of females
total_female = st.number_input("Total Number of Females", min_value=0)

# Input for total number of males
total_male = st.number_input("Total Number of Males", min_value=0)

# Input for purpose
purpose = st.selectbox("Purpose", [
    "Leisure and Holidays", "Visiting Friends and Relatives", "Business",
    "Meetings and Conference", "Volunteering", "Scientific and Academic", "Other"
])

# Input for main activity
main_activity = st.selectbox("Main Activity", [
    "Wildlife tourism", "Cultural tourism", "Mountain climbing", "Beach tourism",
    "Conference tourism", "Hunting tourism", "Bird watching", "Business", "Diving and Sport Fishing"
])

# Input for tour arrangement
tour_arrangement = st.selectbox("Tour Arrangement", ["Independent", "Package Tour"])

# Input for package_transport_international
package_transport_international = st.selectbox("Package Transport International", ["No", "Yes"])

# Input for package_food
package_food = st.selectbox("Package Food", ["No", "Yes"])

# Input for package_transport_local
package_transport_local = st.selectbox("Package Transport Local", ["No", "Yes"])

# Input for package_sightseeing
package_sightseeing = st.selectbox("Package Sightseeing", ["No", "Yes"])

# Input for package_guided_tour
package_guided_tour = st.selectbox("Package Guided Tour", ["No", "Yes"])

# Input for package_insurance
package_insurance = st.selectbox("Package Insurance", ["No", "Yes"])

# Input for nights_stayed
nights_stayed = st.number_input("Nights Stayed", min_value=0)

# Input for payment_mode
payment_mode = st.selectbox("Payment Mode", ["Cash", "Credit Card", "Other", "Travellers Cheque"])

# Input for first_trip
first_trip = st.selectbox("First Trip", ["No", "Yes"])

# Input for most_impressing
most_impressing = st.text_input("Most Impressions")

# Button to make prediction
if st.button("Make Prediction"):
    # Prepare input data
    input_data = {
        "country": country, "age_group": age_group, "travel_with": travel_with,
        "total_female": total_female, "total_male": total_male,
        "purpose": purpose, "main_activity": main_activity,
        "tour_arrangement": tour_arrangement,
        "package_transport_international": package_transport_international,
        "package_food": package_food, "package_transport_local": package_transport_local,
        "package_sightseeing": package_sightseeing, "package_guided_tour": package_guided_tour,
        "package_insurance": package_insurance, "nights_stayed": nights_stayed,
        "payment_mode": payment_mode, "first_trip": first_trip,
        "most_impressing": most_impressing
    }
    # Now you can use input_data to make predictions with your model


# Load the model

# Load the model
model_file_path = ("V:\\Git_repo\\utalii\\xgb_model.pkl")

if os.path.exists(model_file_path):
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please upload a valid model file.")

# If form submitted
if st.button("Make Prediction"):
    data = pd.DataFrame(input_data, index=[0])

    # Factorize object columns
    for colname in data.select_dtypes("object"):
        data[colname] = data[colname].factorize()[0]

    # Perform prediction
    prediction = model.predict(data)

    # Display results
    st.header("Results")
    st.write("Estimated expenditure: ${:.2f}".format(prediction[0]))
