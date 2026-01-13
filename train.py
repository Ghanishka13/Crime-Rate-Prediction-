from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from preprocess import load_data, preprocess_data, split_data, scale_data

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return mse, r2

def save_model(model, scaler, feature_names, filepath='models/crime_rate_model.pkl'):
    """Save the trained model and scaler."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names}, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Use absolute paths to allow running from anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'crime_data.csv')
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        
        model = train_model(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test)
        
        feature_names = X_train.columns.tolist()
        model_path = os.path.join(base_dir, 'models', 'crime_rate_model.pkl')
        save_model(model, scaler, feature_names, filepath=model_path)
    else:
        print(f"Data file not found at {data_path}. Please place your dataset in data/crime_data.csv")