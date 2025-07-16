from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    form_data = {
        'age': 50,
        "sex": "Male",
        "chest_pain": "typical angina",
        'blood_pressure': 120,
        "blodd_sugar": True,
        "ecg": "normal",
        'max_heart_rate': 150,
        "exe_angina": True,
        'st_depression': 2.0,
        "st_slope": "downsloping"
    }

    if request.method == "POST":
        age = request.form["age"]
        form_data['age'] = request.form.get("age",50)

        sex_input = request.form["sex"]
        sex = 1.0 if sex_input=='Male' else 0.0
        form_data['sex'] = request.form.get("sex", "Male")

        chest_pain_input = request.form["chest_pain"]
        chest_pain_map = {
            'typical angina': 0,
            'asymptomatic': 1,
            'non-anginal': 2,
            'atypical angina': 3
        }
        chest_pain = chest_pain_map.get(chest_pain_input, 0)  # Default to 0
        form_data['chest_pain'] = request.form.get("chest_pain", "typical angina")

        blood_pressure = request.form["blood_pressure"]
        form_data['blood_pressure'] = request.form.get("blood_pressure", 145)

        blood_sugar = 1 if 'blood_sugar' in request.form else 0
        form_data['blood_sugar'] = "blood_sugar" in request.form

        ecg_input = request.form["ecg"]
        ecg_map = {
            'lv hypertrophy':0, 
            'normal':1, 
            'st-t abnormality':2
        }
        ecg = chest_pain_map.get(ecg_input, 0)  # Default to 0
        form_data['ecg'] = request.form.get("ecg", "normal")

        max_heart_rate = request.form["max_heart_rate"]
        form_data['max_heart_rate'] = request.form.get("max_heart_rate","")

        exe_angina = 1 if 'exe_angina' in request.form else 0
        form_data['exe_angina'] = "exe_angina" in request.form

        st_depression = request.form["st_depression"]
        form_data['st_depression'] = request.form.get("st_depression","")

        st_slope_input = request.form["st_slope"]
        st_slope_map = {
            'downsloping':0, 
            'flat':1, 
            'upsloping':2
        }
        st_slope = st_slope_map.get(st_slope_input, 0)  # Default to 0
        form_data['st_slope'] = request.form.get("st_slope", "downsloping")

        X = np.array([[float(age), float(sex), float(chest_pain), 
                       float(blood_pressure), float(blood_sugar),
                        float(ecg), float(max_heart_rate), float(exe_angina),
                        float(st_depression), float(st_slope)]])
        pred = model.predict_proba(X)[0][1]
    return render_template("index.html", form_data=form_data, pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)