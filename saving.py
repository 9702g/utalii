import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn import metrics

# Load the dataset
data = pd.read_csv("V:\\Kenyan data\\Kenyan_data.csv")

# Data preprocessing
data['travel_with'] = data['travel_with'].fillna('Alone')
data['total_female'] = data['total_female'].fillna(1.0)
data['total_male'] = data['total_male'].fillna(1.0)
data['most_impressing'] = data['most_impressing'].fillna('No comments')
data['age_group'] = data['age_group'].replace('24-Jan', '1-24')

data["total_female"] = data['total_female'].astype('int')
data["total_male"] = data['total_male'].astype('int')
data["nights_stayed"] = data['nights_stayed'].astype('int')
data["total_people"] = data["total_female"] + data["total_male"]

data.country = data.country.replace(["SWIZERLAND", "BURGARIA", "MALT"], ["SWITZERLAND", "BULGARIA", "MALTA"])
data.country = data.country.replace(["DRC", "SCOTLAND", "UAE", "PHILIPINES", "DJIBOUT", "MORROCO"],
                                    ["DEMOCRATIC REPUBLIC OF THE CONGO", "UNITED KINGDOM", "UNITED ARAB EMIRATES",
                                     "PHILIPPINES", "DJIBOUTI", "MOROCCO"])
data.info_source = data.info_source.replace("Tanzania Mission Abroad", "Kenya Tourist Board")
data.age_group = data.age_group.replace("Jan-24", "1-24")

# Check for missing values
print(data.isnull().sum())


# Encoding categorical variables
for colname in data.select_dtypes("object"):
    data[colname], _ = data[colname].factorize()

# Model Building
import warnings
warnings.filterwarnings('ignore')

# Define features and target
x = data.drop(['total_cost'], axis=1)
y = data['total_cost']

# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Model Training - XGBoost Regressor
model = XGBRegressor()
model.fit(X=x_train, y=y_train)

# Evaluation on training data
training_data_prediction = model.predict(x_train)
score_1 = metrics.r2_score(y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)
print('Training Data - R squared error:', score_1)
print('Training Data - Mean absolute Error:', score_2)

# Evaluation on test data
test_data_prediction = model.predict(x_test)
score_1 = metrics.r2_score(y_test, test_data_prediction)
score_2 = metrics.mean_absolute_error(y_test, test_data_prediction)
print('Test Data - R squared error:', score_1)
print('Test Data - Mean absolute Error:', score_2)
import pickle

# Save the trained model to a .pkl file using pickle
model_file_path = "V:\\my project\\xgb_model.pkl"
with open(model_file_path, 'wb') as file:
    pickle.dump(model, file)

print("Model saved as", model_file_path)
