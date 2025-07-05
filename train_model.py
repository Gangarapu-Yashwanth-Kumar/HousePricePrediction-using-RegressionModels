# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os # Added for path manipulation

def train_and_evaluate_housing_models(data_path='USA_Housing.csv', output_dir='models'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    try:
        data = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure the dataset is in the correct location.")
        return

    # Preprocessing
    X = data.drop(['Price', 'Address'], axis=1)
    y = data['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'RobustRegression': HuberRegressor(),
        'RidgeRegression': Ridge(),
        'LassoRegression': Lasso(),
        'ElasticNet': ElasticNet(),
        'PolynomialRegression': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'SGDRegressor': SGDRegressor(),
        'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=0), # Added random_state for reproducibility
        'RandomForest': RandomForestRegressor(random_state=0), # Added random_state for reproducibility
        'SVM': SVR(),
        'LGBM': lgb.LGBMRegressor(random_state=0), # Added random_state for reproducibility
        'XGBoost': xgb.XGBRegressor(random_state=0), # Added random_state for reproducibility
        'KNN': KNeighborsRegressor()
    }

    # Train and Evaluate Models
    results = []
    trained_model_paths = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        })

        # Save each trained model to the specified output directory
        model_file_path = os.path.join(output_dir, f'{name}.pkl')
        try:
            with open(model_file_path, 'wb') as f:
                pickle.dump(model, f)
            trained_model_paths.append(model_file_path)
            print(f"Model '{name}' saved to {model_file_path}")
        except Exception as e:
            print(f"Error saving model '{name}': {e}")


    # Convert Results to DataFrame and Save to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, 'Model_Evaluation_Results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Evaluation Results saved to {results_csv_path}")

    print("\nTraining and evaluation complete.")
    print("Generated files:")
    for p in trained_model_paths:
        print(f"- {p}")
    print(f"- {results_csv_path}")

if __name__ == "__main__":
    train_and_evaluate_housing_models()