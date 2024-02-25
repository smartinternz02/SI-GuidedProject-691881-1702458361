from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('C:/Users/nikhi/OneDrive/Desktop/varshini Major  project/Training/lrmodel.pkl', 'rb'))


# Define routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/innerpage')
def innerpage():
    return render_template('innerpage.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    # Retrieve form values and convert them to floats
    protein = float(request.form['protein'])
    fat = float(request.form['fat'])
    vitamin_c = float(request.form['vitamin_c'])
    fiber = float(request.form['fiber'])
       
    # Use the model to make predictions
    features = np.array([[protein, fat, vitamin_c, fiber]])
    prediction = model.predict(features)
    output = prediction[0]
    
    return render_template('output.html', prediction_text='The given amount of fat, vitamin c, Fiber will come under the cluster  {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
