# Water Quality Analyser - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 2: Run the Pipeline

```powershell
python main.py
```

This will:

- Generate 1000 training samples and 200 test samples
- Train 6 different ML models
- Create visualizations
- Save all models and metrics

### Step 3: Make Predictions

```powershell
python predict.py
```

Then enter your sensor values when prompted!

## 📊 What You'll Get

After running `main.py`, you'll have:

1. **Trained Models** (in `models/` folder)

   - 6 different regression models
   - Preprocessor for data scaling
   - Performance metrics in JSON

2. **Visualizations** (in `visualizations/` folder)

   - Data distribution plots
   - Correlation heatmap
   - Model comparison charts
   - Prediction accuracy plots

3. **Datasets** (in `data/` folder)
   - Training data (1000 samples)
   - Test data (200 samples)

## 🎯 Example Predictions

### Good Quality Water

```
pH: 7.2
Turbidity: 5.0
Dissolved Oxygen: 8.5
Conductivity: 450
Temperature: 22
TDS: 280
Hardness: 150
Chlorides: 95

→ WQI: ~85 (Good)
```

### Poor Quality Water

```
pH: 8.5
Turbidity: 35.0
Dissolved Oxygen: 4.2
Conductivity: 1200
Temperature: 30
TDS: 750
Hardness: 380
Chlorides: 280

→ WQI: ~35 (Bad)
```

## 🔍 Understanding Your Results

### WQI Scale

- **90-100**: Excellent ✅ - Safe for drinking
- **70-90**: Good 👍 - Suitable for most uses
- **50-70**: Medium ⚠️ - Treatment needed
- **25-50**: Bad ❌ - Not safe
- **0-25**: Very Bad 🚫 - Severe contamination

### Model Performance

Best model is typically **Gradient Boosting** with:

- R² Score: ~0.95 (95% accuracy)
- RMSE: ~4.5 (average error of 4.5 WQI points)

## 💡 Tips

1. **For best results**: Use sensor values within normal ranges
2. **Check visualizations**: They show data patterns and model performance
3. **Compare models**: See `model_metrics.json` for detailed comparison
4. **Batch predictions**: Use CSV files for multiple predictions

## 🆘 Troubleshooting

**Error: "Model not found"**

- Run `python main.py` first to train models

**Error: "Invalid input"**

- Ensure sensor values are within valid ranges
- Check that all 8 parameters are provided

**Poor predictions**

- Check if input values are realistic
- Review the correlation heatmap for parameter relationships

## 📈 Next Steps

1. ✅ Run the pipeline
2. ✅ View visualizations
3. ✅ Make test predictions
4. 🎯 Try with your own sensor data
5. 🔧 Customize models and parameters
6. 🌐 Build a web interface (optional)

---

**Ready to start? Run:** `python main.py`
