import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df, target_column='crime_rate'):
    """Preprocess the data: handle missing values, encode categorical variables, etc."""
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    
    # Fill missing values for features
    df = df.fillna(df.mean(numeric_only=True))
    
    # Encode categorical variables if any
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def split_data(df, target_column='crime_rate', test_size=0.2):
    """Split data into train and test sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_data(X_train, X_test):
    """Scale the features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Example usage
    # Use absolute paths to allow running from anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'crime_data.csv')
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        print("Data preprocessing completed.")
    else:
        print(f"Data file not found at {data_path}. Please place your dataset in data/crime_data.csv")