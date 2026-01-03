"""
Main Execution Script
Runs the complete Water Quality Analyser pipeline
"""

import os
import sys

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created")

def generate_data():
    """Generate training and test data"""
    print("\n" + "=" * 70)
    print("STEP 1: Generating Water Quality Data")
    print("=" * 70)
    from data_generator import generate_water_quality_data
    
    # Generate training data
    train_data = generate_water_quality_data(n_samples=1000, random_state=42)
    train_data.to_csv('data/water_quality_train.csv', index=False)
    print(f"✓ Training data generated: {len(train_data)} samples")
    
    # Generate test data
    test_data = generate_water_quality_data(n_samples=200, random_state=123)
    test_data.to_csv('data/water_quality_test.csv', index=False)
    print(f"✓ Test data generated: {len(test_data)} samples")
    
    print(f"\nQuality distribution in training data:")
    print(train_data['quality_category'].value_counts())

def train_models():
    """Train all ML models"""
    print("\n" + "=" * 70)
    print("STEP 2: Training Machine Learning Models")
    print("=" * 70)
    
    import pandas as pd
    from preprocessing import prepare_data
    from train_models import WQIModelTrainer
    
    # Load data
    df = pd.read_csv('data/water_quality_train.csv')
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(df)
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    # Train models
    trainer = WQIModelTrainer()
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_models('models')
    
    # Get best model
    best_name, _, best_metrics = trainer.get_best_model(metric='r2')
    print(f"\n✓ Best performing model: {best_name}")

def create_visualizations():
    """Create all visualizations"""
    print("\n" + "=" * 70)
    print("STEP 3: Creating Visualizations")
    print("=" * 70)
    
    from visualizations import create_all_visualizations
    create_all_visualizations()
    print("✓ All visualizations created")

def evaluate_on_test_set():
    """Evaluate best model on test set"""
    print("\n" + "=" * 70)
    print("STEP 4: Evaluating on Test Set")
    print("=" * 70)
    
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from preprocessing import WaterQualityPreprocessor
    import joblib
    
    # Load test data
    test_data = pd.read_csv('data/water_quality_test.csv')
    
    # Load preprocessor and best model
    preprocessor = WaterQualityPreprocessor.load('models/preprocessor.pkl')
    model = joblib.load('models/gradient_boosting.pkl')
    
    # Prepare features
    feature_columns = [
        'pH', 'turbidity', 'dissolved_oxygen', 'conductivity',
        'temperature', 'tds', 'hardness', 'chlorides'
    ]
    X_test = test_data[feature_columns]
    y_test = test_data['wqi']
    
    # Preprocess and predict
    X_test_scaled = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\nTest Set Performance:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Create prediction visualization
    from visualizations import WaterQualityVisualizer
    visualizer = WaterQualityVisualizer()
    visualizer.plot_predictions(y_test, y_pred, model_name='Gradient Boosting',
                                save_path='visualizations/test_predictions.png')
    print("\n✓ Test set evaluation completed")

def run_complete_pipeline():
    """Run the complete pipeline"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WATER QUALITY ANALYSER PIPELINE" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Step 1: Create directories
        create_directories()
        
        # Step 2: Generate data
        generate_data()
        
        # Step 3: Train models
        train_models()
        
        # Step 4: Create visualizations
        create_visualizations()
        
        # Step 5: Evaluate on test set
        evaluate_on_test_set()
        
        # Success message
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  - data/water_quality_train.csv")
        print("  - data/water_quality_test.csv")
        print("  - models/*.pkl (trained models)")
        print("  - models/model_metrics.json")
        print("  - visualizations/*.png")
        print("\nNext Steps:")
        print("  1. View visualizations in the 'visualizations' folder")
        print("  2. Run predictions: python predict.py")
        print("  3. Check model metrics in models/model_metrics.json")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_complete_pipeline()
