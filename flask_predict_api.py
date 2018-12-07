"""
Created on Fri Mar 16 21:06:35 2018

@author: Ivana Hybenoa
"""

import pickle
from flask import Flask, request, make_response, send_file
from flasgger import Swagger
import numpy as np
import pandas as pd
import zipfile
import time
from io import BytesIO


with open('./model/final_model.pkl', 'rb') as model_file:
     model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_file', methods=["POST"])
def predict_file():
    """Example file endpoint returning a prediction
    ---
    parameters:
      - name: file_input_test
        in: formData
        type: file
        required: true
    responses:
        200:
            description: OK
    """
    dataset = pd.read_csv(request.files.get("file_input_test"))
    dataset_user = dataset['user']
    # Data preprocessing
    # Creating new X_train with only picked predictors columns
    predictors = ["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "deposits"]
    dataset = dataset[predictors]
    
	
    prediction = model.predict(dataset)
    prediction_probability = model.predict_proba(dataset)
    prediction_probability = prediction_probability[:,1]
    dataset['predicted_class'] = pd.DataFrame(prediction)
    dataset['probability'] = pd.DataFrame(prediction_probability)
    dataset['user'] = pd.DataFrame(dataset_user)
    data = dataset[['user', 'probability', 'predicted_class']]
   
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='predictions', 
                        encoding='utf-8', index=False)
    writer.save()
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['predictions.xlsx'] # names = ['iris_predictions.xlsx', 'file2']
        files = [output]  # files = [output, output2]
        for i in range(len(files)):
            input_data = zipfile.ZipInfo(names[i])
            input_data.date_time = time.localtime(time.time())[:6]
            input_data_compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(input_data, files[i].getvalue())
    memory_file.seek(0)
    
    response = make_response(send_file(memory_file, attachment_filename = 'predictions.zip',
                                       as_attachment=True))
    response.headers['Content-Disposition'] = 'attachment;filename=predictions.zip'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    