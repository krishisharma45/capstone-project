from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import sys

print('This is standard output', file=sys.stdout)

app = Flask(__name__)

def model(input_list):
    input = np.array(input_list).reshape(1, 10)
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    result = loaded_model.predict(input)
    return result[0]

def validate_fields(input_list):
    for value in input_list:
        if value is '':
            return False
        if value.isalpha():
            return False
    return True

def validate_values(input_list):
    for value in input_list:
        if value < 0:
            return False
    return True

@app.route('/')
def display_form():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def display_prediction():
    if request.method == 'POST':
        input_list = request.form.to_dict()
        input_list = list(input_list.values())
        if validate_fields(input_list) == False:
            return render_template('error.html')
        else:
            input_list = list(map(float, input_list))
            if validate_values(input_list) == False:
                return render_template('error.html')
            else:
                result = model(input_list)
                if int(result) == 1:
                    prediction = 'Extremely Vulnerable'
                elif int(result) == 2:
                    prediction = 'Highly Vulnerable'
                elif int(result) == 3:
                    prediction = 'Somewhat Vulnerable'
                else:
                    prediction = 'Not Vulnerable'
                return render_template('results.html', prediction = prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
