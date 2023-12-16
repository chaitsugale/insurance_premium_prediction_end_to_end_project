from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from india_insurance.pipeline.predict_pipeline import CustomData,PredictPipeline

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods= ['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age = request.form.get('age'), 
            sex = request.form.get('sex'),
            bmi = request.form.get('bmi'),
            children = request.form.get('children'),
            smoker =  request.form.get('smoker'),
            region  = request.form.get('region'),
            expenses  = request.form.get('expenses'),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results = results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug = True)
    