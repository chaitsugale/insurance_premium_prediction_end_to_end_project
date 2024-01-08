import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

india_insurance_model = pickle.load(open('india_insurance_best_model.pkl','rb'))
india_insurance_target_encoder = pickle.load(open('india_insurance_target_encoder.pkl','rb'))
india_insurance_transformer = pickle.load(open('india_insurance_transformer.pkl','rb'))

st.title("	:moneybag: :blue[India Insurance Premium Predictor] :moneybag:")
age = st.text_input('Enter Age',18)
age = int(age)


sex = st.selectbox('Select Gender',
                    ('male','female'))

bmi = st.text_input('Enter BMI',18)
bmi = float(bmi)

children = st.select_slider(
    'How Many Childrens do you have?',
    options=[0, 1, 2, 3, 4, 5, 6])
children = int(children)

smoker = st.selectbox('Are you a smoker?',
                       ('yes','no'))

region = st.selectbox('Please select your region',
                          ('southwest','southeast','northeast','northwest'))

insurance_dict = {}

insurance_dict['age'] = age
insurance_dict['sex'] = sex
insurance_dict['bmi'] = bmi
insurance_dict['children'] = children
insurance_dict['smoker'] = smoker
insurance_dict['region'] = region

df = pd.DataFrame(insurance_dict,index = [0])

df['region'] = india_insurance_target_encoder.transform(df['region'])
df['sex'] = df['sex'].map({'male':1, 'female':0})
df['smoker'] = df['smoker'].map({'yes':1, 'no':0})

df =india_insurance_transformer.transform(df)

y_pred = india_insurance_model.predict(df)

if st.button("Please click for the Insurance Premium Price"):
    st.header(f" Your Insurance Premium Price will be : :green[{round(y_pred[0],2)}] INR")
