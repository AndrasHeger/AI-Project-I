from flask import Flask, request, render_template, jsonify, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'mlp_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            # Collect data for multiple days
            num_days = int(request.form['num_days'])
            days_data = []

            for i in range(num_days):
                day_data = {
                    'Age': int(request.form[f'Age_{i}']),
                    'Gender': int(request.form[f'Gender_{i}']),
                    'Bedtime': float(request.form[f'Bedtime_{i}']),
                    'Wakeup_time': float(request.form[f'Wakeup_time_{i}']),
                    'Sleep_duration': float(request.form[f'Sleep_duration_{i}']),
                    'Deep_sleep_percentage': float(request.form[f'Deep_sleep_percentage_{i}']),
                    'Light_sleep_percentage': float(request.form[f'Light_sleep_percentage_{i}']),
                    'Awakenings': int(request.form[f'Awakenings_{i}']),
                    'Alcohol': float(request.form[f'Alcohol_{i}']),
                    'Smoking_status': int(request.form[f'Smoking_status_{i}']),
                    'Exercise_frequency': int(request.form[f'Exercise_frequency_{i}'])
                }
                days_data.append(day_data)

            # Prepare the input features for all days
            input_features = np.array([
                [
                    day['Age'],
                    day['Gender'],
                    day['Bedtime'],
                    day['Wakeup_time'],                   
                    day['Sleep_duration'],
                    day['Deep_sleep_percentage'],
                    day['Light_sleep_percentage'],
                    day['Awakenings'], 
                    day['Alcohol'],
                    day['Smoking_status'],
                    day['Exercise_frequency']
                ]
                for day in days_data
            ])

            

            # Predict sleep quality percentages for all days
            sleep_quality_predictions = model.predict(input_features)

            # Calculate the overall sleep quality percentage
            overall_percentage = (np.mean(sleep_quality_predictions)) *100 # Average quality over all days

            return render_template(
                'result.html',
                overall_percentage=f"{overall_percentage:.2f}%"  # Format as percentage
            )
        except Exception as e:
            return jsonify({'error': str(e)})            

if __name__ == '__main__':
    app.run(debug=True)