
import flask
from flask import request, url_for, redirect
import pickle
import numpy as np
import pandas as pd


# Initialize the app

app = flask.Flask(__name__)

#Random Forest Marijuana Model 
with open("./pickle_files/rf_weed.pkl", "rb") as f:
    rf_weed_pickle = pickle.load(f)

# If you go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up (smiley face)

@app.route("/")
def hello():
    return ":p"

# Let's turn this into an API where you can post input data and get
# back output data after some calculations.

@app.route("/predict", methods=["POST", "GET"])
def predict():

    x_input = []
    for i in range(len(rf_weed_pickle.feature_names)):
        # f_value = 0
        f_value = float(
            request.args.get(rf_weed_pickle.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs = rf_weed_pickle.predict_proba([x_input]).flat

    #Display class with higher probability first
    pred_str = ""
    for class_i in np.argsort(pred_probs)[::-1]:
        pred_str += f"""{rf_weed_pickle.target_names[class_i]}: \
        {str(np.round(pred_probs[class_i]*100, 2))+"%"}<br>"""


    personality_names = ['Neuroticism', 'Extraversion', 'Openness', 
                     'Agreeableness', 'Conscientiousness', 'Impulsiveness',
                      'Sensation-Seeking']
    output_df = pd.DataFrame({"Personality Traits": personality_names, "Scores": x_input})
    output_df.set_index(['Personality Traits'], inplace=True)
    output_df.index.name = None

    # Return a response with a json in it
    # flask has a quick function for that that takes a dict
    return flask.render_template('drug_predictor.html',
    feature_names=rf_weed_pickle.feature_names,
    x_input=output_df.to_html(classes='scores'),
    prediction=pred_str
    )


#Linear SVM Nicotine Pickle
with open("./pickle_files/lin_svm_nic.pkl", "rb") as f:
    lin_svm_nic_pickle = pickle.load(f)

#Nicotine Predictor App
#Lives at /nictoine URL
@app.route("/nicotine", methods=["POST", "GET"])
def nicotine():

    x_input = []
    for i in range(len(lin_svm_nic_pickle.feature_names)):
        # f_value = 0
        f_value = float(
            request.args.get(lin_svm_nic_pickle.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs_nic = lin_svm_nic_pickle.predict_proba([x_input]).flat

    pred_str = ""
    for class_i in np.argsort(pred_probs_nic)[::-1]:
        pred_str += f"""{lin_svm_nic_pickle.target_names[class_i]}: \
        {str(np.round(pred_probs_nic[class_i]*100, 2))+"%"}<br>"""

    personality_names = ['Neuroticism', 'Extraversion', 'Openness', 
                     'Agreeableness', 'Conscientiousness', 'Impulsiveness',
                      'Sensation-Seeking']
    output_df = pd.DataFrame({"Personality Traits": personality_names, "Scores": x_input})
    output_df.set_index(['Personality Traits'], inplace=True)
    output_df.index.name = None

    # Return a response with a json in it
    # flask has a quick function for that that takes a dict
    return flask.render_template('nicotine_predictor.html',
    feature_names=lin_svm_nic_pickle.feature_names,
    x_input=output_df.to_html(classes='scores'),
    prediction=pred_str
    )

#pickle model - Linear SVM Cocaine
with open("./pickle_files/lin_svm_coke.pkl", "rb") as f:
    lin_svm_coke = pickle.load(f)

#Cocaine Predictor App
#Lives at /cocaine URL
@app.route("/cocaine", methods=["POST", "GET"])
def cocaine():

    x_input = []
    for i in range(len(lin_svm_coke.feature_names)):
        # f_value = 0
        f_value = float(
            request.args.get(lin_svm_coke.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs_coke = lin_svm_coke.predict_proba([x_input]).flat

    pred_str = ""
    for class_i in np.argsort(pred_probs_coke)[::-1]:
        pred_str += f"""{lin_svm_coke.target_names[class_i]}: \
        {str(np.round(pred_probs_coke[class_i]*100, 2))+"%"}<br>"""

    personality_names = ['Neuroticism', 'Extraversion', 'Openness', 
                     'Agreeableness', 'Conscientiousness', 'Impulsiveness',
                      'Sensation-Seeking']
    output_df = pd.DataFrame({"Personality Traits": personality_names, "Scores": x_input})
    output_df.set_index(['Personality Traits'], inplace=True)
    output_df.index.name = None

    # Return a response with a json in it
    # flask has a quick function for that that takes a dict
    return flask.render_template('cocaine_predictor.html',
    feature_names=lin_svm_coke.feature_names,
    x_input=output_df.to_html(classes='scores'),
    prediction=pred_str
    )

# For public web serving:
if __name__ == '__main__':
    app.run(host='0.0.0.0')