from flask import Flask, request, jsonify
from flask_cors import CORS
from main_project import predict_risk

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check for required fields
        required_fields = ['bmi', 'heart_rate', 'symptoms']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        # Extract features
        bmi = float(data['bmi'])
        heart_rate = float(data['heart_rate'])
        symptoms = data['symptoms']

        # Make prediction
        # Use dummy temperature for model compatibility
        risk_level = predict_risk(bmi, heart_rate, 36.8)
        risk_levels = ['Low Risk', 'Moderate Risk', 'High Risk']
        main_prediction = risk_levels[risk_level]

        # Map symptoms to possible diseases
        possible_diseases = map_symptoms_to_diseases(symptoms)

        # Create detailed response
        response = {
            'prediction': main_prediction,
            'recommendations': get_recommendations(main_prediction, heart_rate, 36.8),
            'possible_diseases': possible_diseases
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500


def get_recommendations(risk_level, heart_rate, temperature):
    recommendations = []
    if heart_rate > 100:
        recommendations.append(
            "Your heart rate is elevated. Consider resting and deep breathing exercises.")
    elif heart_rate < 60:
        recommendations.append(
            "Your heart rate is lower than normal. Monitor for dizziness or fatigue.")
    if temperature > 37.5:
        recommendations.append(
            "You have a slight fever. Stay hydrated and monitor your temperature.")
    elif temperature > 38:
        recommendations.append(
            "You have a fever. Consider seeking medical attention if it persists.")
    if risk_level == "High Risk":
        recommendations.extend([
            "Please seek immediate medical attention.",
            "Continue monitoring your vital signs.",
            "Avoid strenuous activities."
        ])
    elif risk_level == "Moderate Risk":
        recommendations.extend([
            "Consider consulting with a healthcare provider.",
            "Monitor your symptoms closely.",
            "Ensure you're well-rested and hydrated."
        ])
    else:  # Low Risk
        recommendations.extend([
            "Continue maintaining healthy habits.",
            "Regular exercise and balanced diet recommended.",
            "Monitor your vital signs periodically."
        ])
    return recommendations


def map_symptoms_to_diseases(symptoms):
    # Simple mapping for demonstration
    disease_map = {
        'Fever': ['Flu', 'COVID-19', 'Malaria'],
        'Cough': ['Common Cold', 'Flu', 'COVID-19', 'Bronchitis'],
        'Runny nose': ['Common Cold', 'Allergy'],
        'Headache': ['Migraine', 'Flu', 'COVID-19'],
        'Sore throat': ['Strep Throat', 'Flu', 'COVID-19'],
        'Fatigue': ['Anemia', 'Flu', 'COVID-19'],
        'Shortness of breath': ['Asthma', 'COVID-19', 'Pneumonia'],
        'Muscle pain': ['Flu', 'COVID-19'],
        'Nausea': ['Food Poisoning', 'Gastritis'],
        'Vomiting': ['Food Poisoning', 'Gastritis'],
        'Diarrhea': ['Food Poisoning', 'Gastroenteritis'],
        'Loss of taste': ['COVID-19'],
        'Loss of smell': ['COVID-19'],
        'Chest pain': ['Heart Attack', 'Angina', 'Pneumonia'],
        'Dizziness': ['Low Blood Pressure', 'Anemia'],
        'Rash': ['Allergy', 'Measles'],
        'Chills': ['Flu', 'Malaria'],
        'Sneezing': ['Allergy', 'Common Cold'],
        'Congestion': ['Common Cold', 'Allergy'],
        'Abdominal pain': ['Gastritis', 'Appendicitis'],
        'Back pain': ['Muscle Strain', 'Kidney Infection'],
    }
    diseases = set()
    for symptom in symptoms:
        for disease in disease_map.get(symptom, []):
            diseases.add(disease)
    return list(diseases) if diseases else ['No common disease found']


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
