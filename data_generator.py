"""
Water Quality Data Generator
Generates synthetic water quality sensor data with realistic parameters
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_wqi(ph, turbidity, do, conductivity, temp, tds, hardness, chlorides):
    """
    Calculate Water Quality Index (WQI) based on sensor parameters
    WQI ranges from 0-100, where:
    - 90-100: Excellent
    - 70-90: Good
    - 50-70: Medium
    - 25-50: Bad
    - 0-25: Very Bad
    """
    # Normalized weights for different parameters
    ph_score = max(0, 100 - abs(ph - 7.0) * 10)  # Optimal pH is 7
    turbidity_score = max(0, 100 - turbidity * 2)  # Lower turbidity is better
    do_score = (do / 8.0) * 100  # Higher DO is better (8 mg/L is ideal)
    conductivity_score = max(0, 100 - abs(conductivity - 500) / 10)  # Optimal ~500 μS/cm
    temp_score = max(0, 100 - abs(temp - 25) * 2)  # Optimal temp ~25°C
    tds_score = max(0, 100 - tds / 10)  # Lower TDS is better
    hardness_score = max(0, 100 - abs(hardness - 150) / 3)  # Optimal ~150 mg/L
    chlorides_score = max(0, 100 - chlorides / 5)  # Lower chlorides is better
    
    # Weighted average
    wqi = (
        ph_score * 0.20 +
        turbidity_score * 0.15 +
        do_score * 0.20 +
        conductivity_score * 0.10 +
        temp_score * 0.10 +
        tds_score * 0.10 +
        hardness_score * 0.10 +
        chlorides_score * 0.05
    )
    
    return max(0, min(100, wqi))  # Clamp between 0 and 100

def generate_water_quality_data(n_samples=1000, random_state=42):
    """
    Generate synthetic water quality data
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with water quality parameters and WQI
    """
    np.random.seed(random_state)
    
    # Generate realistic water quality parameters
    data = {
        'pH': np.random.normal(7.2, 0.8, n_samples).clip(5.0, 9.0),
        'turbidity': np.random.exponential(10, n_samples).clip(0, 50),
        'dissolved_oxygen': np.random.normal(7.0, 2.0, n_samples).clip(2, 14),
        'conductivity': np.random.normal(500, 200, n_samples).clip(100, 2000),
        'temperature': np.random.normal(25, 5, n_samples).clip(10, 35),
        'tds': np.random.normal(300, 150, n_samples).clip(50, 1000),
        'hardness': np.random.normal(150, 60, n_samples).clip(50, 500),
        'chlorides': np.random.normal(100, 50, n_samples).clip(10, 400),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate WQI for each sample
    df['wqi'] = df.apply(lambda row: calculate_wqi(
        row['pH'], 
        row['turbidity'], 
        row['dissolved_oxygen'],
        row['conductivity'],
        row['temperature'],
        row['tds'],
        row['hardness'],
        row['chlorides']
    ), axis=1)
    
    # Add noise to make it more realistic
    df['wqi'] += np.random.normal(0, 2, n_samples)
    df['wqi'] = df['wqi'].clip(0, 100)
    
    # Add timestamps
    start_date = datetime(2024, 1, 1)
    df['timestamp'] = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Add water quality category
    def categorize_wqi(wqi):
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
    
    df['quality_category'] = df['wqi'].apply(categorize_wqi)
    
    return df

if __name__ == "__main__":
    # Generate training data
    print("Generating water quality training data...")
    train_data = generate_water_quality_data(n_samples=1000, random_state=42)
    train_data.to_csv('data/water_quality_train.csv', index=False)
    print(f"Training data saved: {len(train_data)} samples")
    print(f"\nData statistics:")
    print(train_data.describe())
    print(f"\nQuality distribution:")
    print(train_data['quality_category'].value_counts())
    
    # Generate test data
    print("\n\nGenerating water quality test data...")
    test_data = generate_water_quality_data(n_samples=200, random_state=123)
    test_data.to_csv('data/water_quality_test.csv', index=False)
    print(f"Test data saved: {len(test_data)} samples")
