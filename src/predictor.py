from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import numpy as np

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        self.feature_columns = list(X.columns)
        
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'model_type': model_type,
            'test_predictions': predictions,
            'test_actual': y_test
        }
    
    def save_model(self, model_path, features_path):
        """Save the trained model and feature columns"""
        joblib.dump(self.model, model_path)
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
    
    def load_model(self, model_path, features_path):
        """Load a saved model and feature columns"""
        self.model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            self.feature_columns = json.load(f)
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("No model trained or loaded. Train or load a model first.")
        return self.model.predict(X)

