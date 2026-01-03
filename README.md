Water Quality Analyser 🌊
An AI-powered water quality analysis system that predicts Water Quality Index (WQI) using machine learning on sensor data. This project demonstrates AI for sustainability by providing accurate water quality assessments from common sensor measurements.

🎯 Project Overview
The Water Quality Analyser uses regression machine learning models to predict the Water Quality Index (WQI) based on multiple water quality parameters collected from sensors. This enables:

Real-time water quality monitoring
Early contamination detection
Automated quality assessment
Support for environmental sustainability
💡 Innovation & Impact
AI for Sustainability
This project addresses UN Sustainable Development Goal 6 (Clean Water and Sanitation) by:

Enabling continuous water quality monitoring
Reducing the need for expensive laboratory tests
Providing early warning systems for water contamination
Supporting data-driven water management decisions
Technical Innovation
Multiple ML Models: Compares 6 different regression algorithms
Comprehensive Analysis: Uses 8 key water quality parameters
Real-time Predictions: Fast inference for immediate results
Interpretable Results: Clear categorization and recommendations
📊 Features
Water Quality Parameters Analyzed
pH - Acidity/alkalinity level
Turbidity - Water clarity (NTU)
Dissolved Oxygen - Oxygen content (mg/L)
Conductivity - Electrical conductivity (μS/cm)
Temperature - Water temperature (°C)
TDS - Total Dissolved Solids (mg/L)
Hardness - Water hardness (mg/L)
Chlorides - Chloride content (mg/L)
Machine Learning Models
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
Gradient Boosting Regressor ⭐ (Best performing)
Support Vector Regression (SVR)
WQI Categories
90-100: Excellent - Safe for all uses
70-90: Good - Suitable for most uses
50-70: Medium - Treatment recommended
25-50: Bad - Significant treatment required
0-25: Very Bad - Severe contamination
🚀 Quick Start
Prerequisites
Python 3.8 or higher
pip package manager
Installation
Clone or navigate to the project directory
cd "c:\Users\KIIT0001\OneDrive\Desktop\AI\water_quality_analyser"
Install required packages
pip install -r requirements.txt
Usage
Option 1: Run Complete Pipeline
python main.py
This will:

Generate synthetic water quality data
Train all ML models
Create visualizations
Evaluate model performance
Save trained models
Option 2: Interactive Prediction
python predict.py
Enter sensor values when prompted to get instant WQI predictions.

Option 3: Batch Prediction from CSV
python predict.py input_data.csv output_predictions.csv
Option 4: Step-by-Step Execution
Generate Data:

python data_generator.py
Train Models:

python train_models.py
Create Visualizations:

python visualizations.py
📁 Project Structure
water_quality_analyser/
│
├── main.py                  # Complete pipeline execution
├── data_generator.py        # Synthetic data generation
├── preprocessing.py         # Data preprocessing utilities
├── train_models.py          # Model training and evaluation
├── predict.py               # Prediction interface
├── visualizations.py        # Visualization tools
├── requirements.txt         # Python dependencies
│
├── data/                    # Generated datasets
│   ├── water_quality_train.csv
│   └── water_quality_test.csv
│
├── models/                  # Trained models
│   ├── *.pkl (model files)
│   ├── preprocessor.pkl
│   └── model_metrics.json
│
└── visualizations/          # Generated plots
    ├── data_distribution.png
    ├── correlation_heatmap.png
    ├── wqi_by_category.png
    ├── model_comparison.png
    └── test_predictions.png
🔬 Technical Details
Data Generation
Uses realistic distributions for water quality parameters
Generates 1000 training samples and 200 test samples
Implements physics-based WQI calculation
Adds realistic noise to simulate sensor variations
Preprocessing
Standard scaling for normalization
Feature engineering (pH deviation, oxygen saturation, etc.)
Train-validation-test split
Reproducible pipeline with sklearn
Model Training
Cross-validation on training set
Hyperparameter optimization
Multiple evaluation metrics (R², RMSE, MAE, MAPE)
Model comparison and selection
Evaluation Metrics
R² Score: Measures prediction accuracy (higher is better)
RMSE: Root Mean Squared Error (lower is better)
MAE: Mean Absolute Error (lower is better)
MAPE: Mean Absolute Percentage Error (lower is better)
📈 Expected Results
Based on synthetic data, typical model performance:

Model	R² Score	RMSE	MAE
Gradient Boosting	~0.95	~4.5	~3.2
Random Forest	~0.94	~5.0	~3.5
Ridge Regression	~0.85	~8.0	~6.0
Note: Actual results may vary based on random seed and data generation

🎨 Visualizations
The project generates several informative visualizations:

Data Distribution: Histograms of all water quality parameters
Correlation Heatmap: Shows relationships between parameters
WQI by Category: Box plots and count distributions
Model Comparison: Performance metrics across all models
Prediction Plots: Actual vs predicted values with residuals
🔧 Customization
Adjust Model Parameters
Edit train_models.py to modify hyperparameters:

'random_forest': RandomForestRegressor(
    n_estimators=100,  # Change number of trees
    max_depth=15,      # Adjust tree depth
    random_state=42
)
Add New Features
Modify preprocessing.py to add custom engineered features:

def add_derived_features(df):
    df['your_feature'] = df['param1'] * df['param2']
    return df
Generate More Data
Change sample size in data_generator.py:

train_data = generate_water_quality_data(n_samples=5000)
🌍 Real-World Applications
Municipal Water Treatment Plants

Continuous monitoring of treated water quality
Automated alerts for quality deviations
Environmental Monitoring

River and lake quality assessment
Pollution tracking and source identification
Industrial Water Management

Process water quality control
Wastewater treatment optimization
Agricultural Applications

Irrigation water quality monitoring
Aquaculture water management
Smart Cities

IoT-based water quality networks
Public health protection systems
🎓 Learning Outcomes
This project demonstrates:

End-to-end ML pipeline development
Regression problem solving
Feature engineering techniques
Model comparison and selection
Data visualization best practices
AI for social good and sustainability
📚 Dependencies
numpy: Numerical computations
pandas: Data manipulation
scikit-learn: Machine learning algorithms
matplotlib: Plotting and visualization
seaborn: Statistical visualizations
joblib: Model serialization
🤝 Contributing
Feel free to:

Add new ML models
Improve feature engineering
Enhance visualizations
Add real sensor data integration
Implement web interface
📝 Future Enhancements
 Web-based dashboard (Flask/Streamlit)
 Real-time sensor data integration
 Time-series forecasting for trends
 Deep learning models (LSTM/Neural Networks)
 Mobile app for field testing
 API for third-party integration
 Multi-location monitoring system
 Alert notification system
🏆 Model Performance Insights
Why Gradient Boosting Performs Best?
Sequential Error Correction: Each tree corrects previous mistakes
Non-linear Relationships: Captures complex interactions between parameters
Robustness: Handles outliers and noise effectively
Feature Importance: Identifies key water quality indicators
Key Findings
pH and dissolved oxygen are strongest predictors
Turbidity and conductivity show high correlation with WQI
Temperature has moderate influence on overall quality
Combined hardness-chloride features improve predictions
💻 Example Usage
from predict import WQIPredictor

# Initialize predictor
predictor = WQIPredictor()

# Prepare sensor data
sensor_data = {
    'pH': 7.2,
    'turbidity': 5.0,
    'dissolved_oxygen': 8.5,
    'conductivity': 450,
    'temperature': 22,
    'tds': 280,
    'hardness': 150,
    'chlorides': 95
}

# Get prediction
result = predictor.predict(sensor_data)

print(f"WQI: {result['wqi']}")
print(f"Quality: {result['quality_category']}")
print(f"Description: {result['description']}")
📞 Support
For questions or issues:

Check the code comments for detailed explanations
Review the generated visualizations for insights
Examine model_metrics.json for performance details
🌟 Acknowledgments
This project demonstrates:

Practical AI application for environmental sustainability
Industry-standard ML practices and workflows
Comprehensive documentation and code quality
Reproducible research with clear methodology
Built with ❤️ for a sustainable future

Water is life. Let's keep it clean with AI! 🌊🌍
