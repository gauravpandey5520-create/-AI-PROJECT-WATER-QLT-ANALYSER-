"""
Water Quality Predictor
Simple interface for predicting Water Quality Index (WQI) from sensor data
"""

import numpy as np
import pandas as pd
import joblib
from preprocessing import WaterQualityPreprocessor
import os

class WQIPredictor:
    """
    Water Quality Index Predictor
    """
    
    def __init__(self, model_path='models/gradient_boosting.pkl',
                 preprocessor_path='models/preprocessor.pkl'):
        """
        Initialize predictor with trained model and preprocessor
        
        Parameters:
        -----------
        model_path : str
            Path to saved model
        preprocessor_path : str
            Path to saved preprocessor
        """
        self.model = joblib.load(model_path)
        self.preprocessor = WaterQualityPreprocessor.load(preprocessor_path)
        self.feature_columns = [
            'pH', 'turbidity', 'dissolved_oxygen', 'conductivity',
            'temperature', 'tds', 'hardness', 'chlorides'
        ]
    
    def predict(self, sensor_data):
        """
        Predict WQI from sensor data
        
        Parameters:
        -----------
        sensor_data : dict or pd.DataFrame
            Dictionary or DataFrame with sensor readings
            Required keys: pH, turbidity, dissolved_oxygen, conductivity,
                          temperature, tds, hardness, chlorides
        
        Returns:
        --------
        dict
            Prediction results with WQI value and quality category
        """
        # Convert to DataFrame if dict
        if isinstance(sensor_data, dict):
            df = pd.DataFrame([sensor_data])
        else:
            df = sensor_data.copy()
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")
        
        # Preprocess data
        df_processed = self.preprocessor.transform(df[self.feature_columns])
        
        # Make prediction
        wqi_pred = self.model.predict(df_processed)[0]
        wqi_pred = max(0, min(100, wqi_pred))  # Clamp between 0 and 100
        
        # Determine quality category
        quality_category = self._categorize_wqi(wqi_pred)
        
        # Get quality description
        description = self._get_quality_description(quality_category)
        
        return {
            'wqi': round(wqi_pred, 2),
            'quality_category': quality_category,
            'description': description,
            'sensor_data': sensor_data if isinstance(sensor_data, dict) else sensor_data.iloc[0].to_dict()
        }
    
    def predict_batch(self, sensor_data_list):
        """
        Predict WQI for multiple sensor readings
        
        Parameters:
        -----------
        sensor_data_list : list of dict or pd.DataFrame
            List of sensor readings
        
        Returns:
        --------
        list of dict
            Prediction results for each input
        """
        if isinstance(sensor_data_list, pd.DataFrame):
            df = sensor_data_list
        else:
            df = pd.DataFrame(sensor_data_list)
        
        # Preprocess
        df_processed = self.preprocessor.transform(df[self.feature_columns])
        
        # Predict
        predictions = self.model.predict(df_processed)
        predictions = np.clip(predictions, 0, 100)
        
        # Create results
        results = []
        for idx, wqi_pred in enumerate(predictions):
            quality_category = self._categorize_wqi(wqi_pred)
            results.append({
                'wqi': round(wqi_pred, 2),
                'quality_category': quality_category,
                'description': self._get_quality_description(quality_category)
            })
        
        return results
    
    @staticmethod
    def _categorize_wqi(wqi):
        """Categorize WQI value"""
        if wqi >= 90:
            return 'Excellent'
        elif wqi >= 70:
            return 'Good'
        elif wqi >= 50:
            return 'Medium'
        elif wqi >= 25:
            return 'Bad'
        else:
            return 'Very Bad'
    
    @staticmethod
    def _get_quality_description(category):
        """Get description for quality category"""
        descriptions = {
            'Excellent': 'Water quality is excellent. Safe for all uses including drinking.',
            'Good': 'Water quality is good. Suitable for most uses with minimal treatment.',
            'Medium': 'Water quality is medium. Treatment recommended before consumption.',
            'Bad': 'Water quality is bad. Significant treatment required. Not suitable for drinking.',
            'Very Bad': 'Water quality is very bad. Severe contamination. Avoid all uses.'
        }
        return descriptions.get(category, 'Unknown quality level')

def interactive_prediction():
    """
    Interactive command-line interface for water quality prediction
    """
    print("=" * 70)
    print("WATER QUALITY INDEX PREDICTOR")
    print("=" * 70)
    print("\nThis tool predicts Water Quality Index (WQI) from sensor data.")
    print("Please enter the following water quality parameters:\n")
    
    try:
        # Get user input
        ph = float(input("pH (5.0-9.0): "))
        turbidity = float(input("Turbidity in NTU (0-50): "))
        dissolved_oxygen = float(input("Dissolved Oxygen in mg/L (2-14): "))
        conductivity = float(input("Conductivity in μS/cm (100-2000): "))
        temperature = float(input("Temperature in °C (10-35): "))
        tds = float(input("Total Dissolved Solids in mg/L (50-1000): "))
        hardness = float(input("Hardness in mg/L (50-500): "))
        chlorides = float(input("Chlorides in mg/L (10-400): "))
        
        # Create sensor data dictionary
        sensor_data = {
            'pH': ph,
            'turbidity': turbidity,
            'dissolved_oxygen': dissolved_oxygen,
            'conductivity': conductivity,
            'temperature': temperature,
            'tds': tds,
            'hardness': hardness,
            'chlorides': chlorides
        }
        
        # Initialize predictor
        predictor = WQIPredictor()
        
        # Make prediction
        result = predictor.predict(sensor_data)
        
        # Display results
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"\nWater Quality Index (WQI): {result['wqi']}")
        print(f"Quality Category: {result['quality_category']}")
        print(f"\nDescription: {result['description']}")
        print("\n" + "=" * 70)
        
    except FileNotFoundError as e:
        print(f"\nError: Model or preprocessor not found.")
        print("Please train the model first by running: python train_models.py")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please ensure all inputs are valid numbers within the specified ranges.")

def predict_from_file(input_file, output_file='predictions.csv'):
    """
    Predict WQI for data from a CSV file
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with sensor data
    output_file : str
        Path to save predictions
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Making predictions...")
    predictor = WQIPredictor()
    results = predictor.predict_batch(df)
    
    # Add predictions to DataFrame
    df['predicted_wqi'] = [r['wqi'] for r in results]
    df['predicted_category'] = [r['quality_category'] for r in results]
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Display summary
    print("\nPrediction Summary:")
    print(df[['predicted_wqi', 'predicted_category']].describe())
    print("\nCategory Distribution:")
    print(df['predicted_category'].value_counts())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Predict from file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict_from_file(input_file, output_file)
    else:
        # Interactive mode
        interactive_prediction()
