from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model_LR.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
	return render_template('bank.html')
        
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
		emp_var_rate = float(request.form.get('emp_var_rate',False))
		cons_price_idx = float(request.form.get('cons_price_idx',False))
		cons_conf_idx = float(request.form.get('cons_conf_idx',False))
		euribor3m = float(request.form.get('euribor3m',False))
		nr_employed = int(request.form.get('nr_employed',False))
		prediction=model.predict([[age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed]])
		output=prediction[0]
		if output==1:
			a = "will"
		else:
			a = "will not"
		
		return render_template('bank.html',prediction_text="This customer {}".format(a) + " subscribe to the term deposit")
		
	else:
		return render_template('bank.html')
		
        
if __name__=="__main__":
	app.run(debug=True, use_reloader=False)
