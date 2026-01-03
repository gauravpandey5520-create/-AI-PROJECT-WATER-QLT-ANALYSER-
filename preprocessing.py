"""
Data Preprocessing Module
Handles data cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

class WaterQualityPreprocessor:
    """
    Preprocessor for water quality data
    """
    
    def __init__(self, scaling_method='standard'):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        scaling_method : str
            'standard' for StandardScaler or 'minmax' for MinMaxScaler
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'wqi'
        
    def fit(self, df):
        """
        Fit the preprocessor on training data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        """
        # Define feature columns (exclude target and non-numeric columns)
        self.feature_columns = [
            'pH', 'turbidity', 'dissolved_oxygen', 'conductivity',
            'temperature', 'tds', 'hardness', 'chlorides'
        ]
        
        # Fit scaler on features
        self.scaler.fit(df[self.feature_columns])
        
        return self
    
    def transform(self, df):
        """
        Transform data using fitted preprocessor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        df_transformed = df.copy()
        
        # Scale features
        df_transformed[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_transformed
    
    def fit_transform(self, df):
        """
        Fit and transform data in one step
        """
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df):
        """
        Inverse transform scaled features back to original scale
        """
        df_inverse = df.copy()
        df_inverse[self.feature_columns] = self.scaler.inverse_transform(df[self.feature_columns])
        return df_inverse
    
    def save(self, filepath):
        """
        Save preprocessor to disk
        """
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'scaling_method': self.scaling_method
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load preprocessor from disk
        """
        data = joblib.load(filepath)
        preprocessor = WaterQualityPreprocessor(scaling_method=data['scaling_method'])
        preprocessor.scaler = data['scaler']
        preprocessor.feature_columns = data['feature_columns']
        return preprocessor

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (X_train, X_val, y_train, y_val, preprocessor)
    """
    # Initialize preprocessor
    preprocessor = WaterQualityPreprocessor(scaling_method='standard')
    
    # Extract features and target
    feature_columns = [
        'pH', 'turbidity', 'dissolved_oxygen', 'conductivity',
        'temperature', 'tds', 'hardness', 'chlorides'
    ]
    
    X = df[feature_columns]
    y = df['wqi']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fit preprocessor on training data
    preprocessor.fit(pd.DataFrame(X_train, columns=feature_columns))
    
    # Transform data
    X_train_scaled = preprocessor.transform(pd.DataFrame(X_train, columns=feature_columns))
    X_val_scaled = preprocessor.transform(pd.DataFrame(X_val, columns=feature_columns))
    
    return X_train_scaled, X_val_scaled, y_train, y_val, preprocessor

def add_derived_features(df):
    """
    Add engineered features to improve model performance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
        
    Returns:
    --------
    pd.DataFrame
        Data with additional features
    """
    df_enhanced = df.copy()
    
    # pH deviation from neutral
    df_enhanced['ph_deviation'] = np.abs(df['pH'] - 7.0)
    
    # Oxygen saturation ratio
    df_enhanced['oxygen_saturation'] = df['dissolved_oxygen'] / 8.0
    
    # Temperature zone (cold, moderate, warm)
    df_enhanced['temp_cold'] = (df['temperature'] < 20).astype(int)
    df_enhanced['temp_warm'] = (df['temperature'] > 28).astype(int)
    
    # Pollution indicators
    df_enhanced['high_turbidity'] = (df['turbidity'] > 20).astype(int)
    df_enhanced['high_conductivity'] = (df['conductivity'] > 800).astype(int)
    
    # Combined hardness-chloride indicator
    df_enhanced['mineral_load'] = df['hardness'] + df['chlorides']
    
    return df_enhanced

if __name__ == "__main__":
    # Test preprocessing
    print("Testing data preprocessing...")
    
    # Load data
    df = pd.read_csv('data/water_quality_train.csv')
    print(f"Loaded {len(df)} samples")
    
    # Prepare data
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(df)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"\nFeatures: {preprocessor.feature_columns}")
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    print("\nPreprocessing test completed successfully!")
