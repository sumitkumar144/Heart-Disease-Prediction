# importing the necessary dependencies
# Code Reference is taken from https://github.com/shobhitsrivastava-ds/ML-MT-WebApp/

from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
import pickle

# initializing a flask app
app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

# route to display the home page
@app.route("/")

@app.route("/home",methods=["GET","POST"])
@cross_origin()
def homePage():
     return render_template("index.html")

@app.route("/about",methods=["GET","POST"])
@cross_origin()
def about():
    return render_template("about.html")

@app.route("/heartpred",methods=["GET","POST"])
@cross_origin()
def heartPred():
    return render_template("heartDisease.html")

# route to show the predictions in a web UI
@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            age=float(request.form['age'])
            gender = int(request.form['gender'])
            cp = int(request.form['cp'])
            restbp = float(request.form['restbp'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            mhrs = float(request.form['mhrs'])
            exia = int(request.form['exia'])
            std = float(request.form['std'])
            slope = int(request.form['slope'])
            ca = float(request.form['ca'])
            Thal = int(request.form['Thal'])

            # Initialize File Name
            model_file = 'modelForPrediction.sav'
            StandardScaler_File = 'standardScaler.sav'

            # loading the model file from the storage
            loaded_model = pickle.load(open(model_file, 'rb'))

            #loading Scaler pickle file
            scaler = pickle.load(open(StandardScaler_File, 'rb'))

            # predictions using the loaded model file and scaler file
            prediction = loaded_model.predict(scaler.transform([[age, gender, cp, restbp, chol, fbs, restecg, mhrs, exia, std, slope, ca, Thal]]))
            print('prediction is', prediction)

            # showing the prediction results in a UI
            if prediction == 0:
                return render_template('predictionNo.html')
            elif prediction == 1:
                return render_template('predictionYes.html')
            else:
                return "something is wrong"

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    # to run locally
    #app.run(host='127.0.0.1', port=8000, debug=True)

    # to run on cloud AWS
    app.run(debug=True)  # running the app

