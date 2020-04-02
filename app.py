# importing the main library for the application
from flask import (Flask,            # the main framework
                  request,           # takes the requests from the forms
                  url_for,           # for path to html and css
                  redirect,          # for redirecting to a different template
                  render_template,   # for rendering the html templates
)

import pickle     # for putting the data on the serial interface so that the predictor app can use this
import numpy as np

# initialise the app
app = Flask(__name__)

# Get the model
model=pickle.load(open('model.pkl','rb'))

# The Home route
@app.route('/')
def hello():
    return render_template("index.html")

# Route to predict values
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        int_features=[]
        int_features.append(int(request.form['temp']))
        int_features.append(int(request.form['oxy']))
        int_features.append(int(request.form['hum']))
        final=[np.array(int_features)]
        print("Initial Features : " + str(int_features))
        print("Numpy features : " + str(final))
        prediction=model.predict_proba(final)

        return render_template('index.html', pred = prediction[0][1])
    else:
        return render_template('index.html')


# Starting the app
if __name__ == '__main__':
    app.run(debug=True)
