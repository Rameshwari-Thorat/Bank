from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model_LR.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
	return render_template('Bank.html')
        
@app.route("/predict", methods=['POST'])
def predict():
	if request.method == 'POST':
		age = int(request.form.get('age',False))
		job = int(request.form.get('job',False))
		marital = int(request.form.get('marital',False))
		education = int(request.form.get('education',False))
		default = int(request.form.get('default',False))
		housing = int(request.form.get('housing',False))
		loan = int(request.form.get('loan',False))
		contact = int(request.form.get('contact',False))
		month = int(request.form.get('month',False))
		day_of_week = int(request.form.get('day_of_week',False))
		duration = int(request.form.get('duration',False))
		campaign = int(request.form.get('campaign',False))
		pdays = int(request.form.get('pdays',False))
		previous = int(request.form.get('previous',False))
		poutcome = int(request.form.get('poutcome',False))
		emp.var.rate = int(request.form.get('emp.var.rate',False))
		cons.price.idx = int(request.form.get('cons.price.idx',False))
		cons.conf.idx = int(request.form.get('cons.conf.idx',False))
		euribor3m = int(request.form.get('euribor3m',False))
		nr.employed = int(request.form.get('nr.employed',False))
		output=prediction[0]
		
		return render_template('Bank.html',prediction_text="You Fall into category:-  {}".format(output))
		
	else:
		return render_template('Bank.html')
		
        
if __name__=="__main__":
	app.run(debug=True, use_reloader=False)