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

st.title("India Insurance Predictor")