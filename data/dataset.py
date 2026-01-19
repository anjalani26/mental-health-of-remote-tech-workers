#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#import the dataset
df = pd.read_csv("data/raw_dataset/mental_health_remote_workers.csv")
print(df.head())
print(df.info())

#check for duplicate rows
print(df[df.duplicated()])
#There are no duplicate rows

#categorical variables are encoded using one hot encoding
df_final = pd.get_dummies(df, columns=['Gender', 'Country', 'Job_Role', 'Work_Mode', 'Mental_Health_Status', 'Exercise_Frequency','Internet_Issues_Frequency'], dtype=int)

#bool type categorical columns cover to int type
df_final['Has_Access_To_Therapist'] = df_final['Has_Access_To_Therapist'].astype(int)
df_final['Willing_To_Return_Onsite'] = df_final['Willing_To_Return_Onsite'].astype(int)

#removed Employee_ID and Name
cols_to_remove = ['Employee_ID', 'Name']
df_final = df_final.drop(columns=cols_to_remove)

#Bunrout Score is taken to the range or 0 to 10
scaler = MinMaxScaler(feature_range=(0,10))
df_final['Burnout_Score'] = scaler.fit_transform(df[['Burnout_Score']]).round(1)

df_final.to_csv('data/cleaned_dataset/encoded_dataset.csv', index=False)
print("File saved as 'encoded_dataset.csv'")

