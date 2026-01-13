import joblib
import pandas as pd
import numpy as np
import os


def load_model(filepath='models/crime_rate_model.pkl'):
    """Load the trained model and scaler."""
    data = joblib.load(filepath)
    return data['model'], data['scaler'], data.get('feature_names', [])


def predict_crime_rate(model, scaler, feature_names, input_data):
    """Predict crime rate for new data."""
    
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features (if any)
    input_df = pd.get_dummies(input_df)

    # Ensure all training features exist
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Predict crime rate
    prediction = model.predict(input_scaled)

    return prediction[0]


def get_user_input():
    """Get crime-related data from terminal."""
    print("\nEnter Area Details:")

    population = int(input("Population: "))
    unemployment_rate = float(input("Unemployment rate (%): "))
    education_level = float(input("Average education level (years): "))

    return {
        'population': population,
        'unemployment_rate': unemployment_rate,
        'education_level': education_level
    }


if __name__ == "__main__":
    # Absolute path to model file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'crime_rate_model.pkl')

    if os.path.exists(model_path):
        model, scaler, feature_names = load_model(model_path)

        print("\n=== Crime Rate Prediction System ===")

        user_data = get_user_input()

        prediction = predict_crime_rate(
            model, scaler, feature_names, user_data
        )

        print(f"\n🔮 Predicted Crime Rate: {prediction}")

    else:
        print(f"❌ Model not found at {model_path}. Please train the model first.")
