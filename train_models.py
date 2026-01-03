"""
Water Quality Index Prediction Models
Implements multiple regression models for WQI prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from datetime import datetime

class WQIModelTrainer:
    """
    Trainer for Water Quality Index prediction models
    """
    
    def __init__(self):
        """
        Initialize model trainer with multiple regression models
        """
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'svr': SVR(kernel='rbf', C=10, gamma='scale')
        }
        
        self.trained_models = {}
        self.performance_metrics = {}
        
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None):
        """
        Train a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation targets
            
        Returns:
        --------
        dict
            Performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred, "Training")
        
        metrics = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, "Validation")
            metrics['val_metrics'] = val_metrics
        
        # Store trained model and metrics
        self.trained_models[model_name] = model
        self.performance_metrics[model_name] = metrics
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all available models
        
        Returns:
        --------
        dict
            Performance metrics for all models
        """
        print("=" * 60)
        print("Training all models...")
        print("=" * 60)
        
        all_metrics = {}
        
        for model_name in self.models.keys():
            metrics = self.train_model(model_name, X_train, y_train, X_val, y_val)
            all_metrics[model_name] = metrics
        
        return all_metrics
    
    def _calculate_metrics(self, y_true, y_pred, dataset_name=""):
        """
        Calculate performance metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        if dataset_name:
            print(f"\n{dataset_name} Metrics:")
            print(f"  MSE:  {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def get_best_model(self, metric='r2'):
        """
        Get the best performing model based on validation metric
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison ('r2', 'rmse', 'mae')
            
        Returns:
        --------
        tuple
            (model_name, model, metrics)
        """
        if not self.performance_metrics:
            raise ValueError("No models have been trained yet")
        
        best_model_name = None
        best_score = float('-inf') if metric == 'r2' else float('inf')
        
        for model_name, metrics in self.performance_metrics.items():
            if 'val_metrics' in metrics:
                score = metrics['val_metrics'][metric]
            else:
                score = metrics['train_metrics'][metric]
            
            # For R², higher is better; for RMSE/MAE, lower is better
            if metric == 'r2':
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
        
        return (
            best_model_name,
            self.trained_models[best_model_name],
            self.performance_metrics[best_model_name]
        )
    
    def save_models(self, directory='models'):
        """
        Save all trained models to disk
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = os.path.join(directory, f'{model_name}.pkl')
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
        
        # Save metrics
        metrics_file = os.path.join(directory, 'model_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        print(f"Saved metrics to {metrics_file}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from disk
        """
        return joblib.load(filepath)
    
    def compare_models(self):
        """
        Print a comparison table of all models
        """
        if not self.performance_metrics:
            print("No models have been trained yet")
            return
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
        print("-" * 80)
        
        for model_name, metrics in self.performance_metrics.items():
            # Use validation metrics if available, otherwise training metrics
            m = metrics.get('val_metrics', metrics.get('train_metrics'))
            
            print(f"{model_name:<25} {m['r2']:<10.4f} {m['rmse']:<10.4f} "
                  f"{m['mae']:<10.4f} {m['mape']:<10.2f}%")
        
        print("=" * 80)
        
        # Highlight best model
        best_name, _, _ = self.get_best_model(metric='r2')
        print(f"\n🏆 Best Model: {best_name}")
        print("=" * 80)

if __name__ == "__main__":
    # Test model training
    print("Testing model training...")
    
    # Load preprocessed data
    from preprocessing import prepare_data
    import pandas as pd
    
    df = pd.read_csv('data/water_quality_train.csv')
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(df)
    
    # Train models
    trainer = WQIModelTrainer()
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_models('models')
    
    print("\nModel training completed successfully!")
