# Crime Rate Prediction System

This project is a machine learning system for predicting crime rates based on various socio-economic and demographic factors.

## Project Structure

- `data/`: Contains datasets used for training and testing the model.
- `src/`: Source code for data preprocessing, model training, and prediction.
- `models/`: Saved trained models.
- `notebooks/`: Jupyter notebooks for data exploration and analysis.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Place your dataset in the `data/` folder.
2. Run the data preprocessing script: `python src/preprocess.py`
3. Train the model: `python src/train.py`
4. Make predictions: `python src/predict.py`

## Dataset

The system expects a CSV file with columns such as population, unemployment rate, education level, etc., and a target column for crime rate.

## Model

Uses Random Forest Regressor for prediction.

## Contributing

Feel free to contribute by improving the model or adding new features.