from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.components.data_ingestion import load_image_data

application = Flask(__name__)
app = application



## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 



if __name__=="__main__":
    df = load_image_data()  
    print(df)  
    app.run(host="0.0.0.0") 
