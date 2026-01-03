"""
Visualization Module
Creates charts and visualizations for water quality analysis and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class WaterQualityVisualizer:
    """
    Visualization tools for water quality analysis
    """
    
    @staticmethod
    def plot_data_distribution(df, save_path='visualizations/data_distribution.png'):
        """
        Plot distribution of water quality parameters
        """
        feature_columns = [
            'pH', 'turbidity', 'dissolved_oxygen', 'conductivity',
            'temperature', 'tds', 'hardness', 'chlorides', 'wqi'
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(feature_columns):
            axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col.replace("_", " ").title()} Distribution')
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved data distribution plot to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_correlation_heatmap(df, save_path='visualizations/correlation_heatmap.png'):
        """
        Plot correlation heatmap of features
        """
        feature_columns = [
            'pH', 'turbidity', 'dissolved_oxygen', 'conductivity',
            'temperature', 'tds', 'hardness', 'chlorides', 'wqi'
        ]
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[feature_columns].corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Heatmap of Water Quality Parameters', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_wqi_by_category(df, save_path='visualizations/wqi_by_category.png'):
        """
        Plot WQI distribution by quality category
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        category_order = ['Very Bad', 'Bad', 'Medium', 'Good', 'Excellent']
        existing_categories = [cat for cat in category_order if cat in df['quality_category'].values]
        
        sns.boxplot(data=df, x='quality_category', y='wqi', order=existing_categories,
                   palette='Set2', ax=ax1)
        ax1.set_title('WQI Distribution by Quality Category', fontsize=14)
        ax1.set_xlabel('Quality Category')
        ax1.set_ylabel('Water Quality Index (WQI)')
        ax1.grid(True, alpha=0.3)
        
        # Count plot
        category_counts = df['quality_category'].value_counts()
        colors = plt.cm.Set2(range(len(category_counts)))
        
        ax2.bar(range(len(category_counts)), category_counts.values, color=colors)
        ax2.set_xticks(range(len(category_counts)))
        ax2.set_xticklabels(category_counts.index, rotation=45)
        ax2.set_title('Sample Count by Quality Category', fontsize=14)
        ax2.set_xlabel('Quality Category')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved WQI category plot to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_model_comparison(metrics_file='models/model_metrics.json',
                             save_path='visualizations/model_comparison.png'):
        """
        Plot comparison of model performance
        """
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract metrics
        model_names = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for model_name, model_metrics in metrics.items():
            m = model_metrics.get('val_metrics', model_metrics.get('train_metrics'))
            model_names.append(model_name.replace('_', ' ').title())
            r2_scores.append(m['r2'])
            rmse_scores.append(m['rmse'])
            mae_scores.append(m['mae'])
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # R² Score
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        axes[0].barh(model_names, r2_scores, color=colors)
        axes[0].set_xlabel('R² Score')
        axes[0].set_title('R² Score Comparison (Higher is Better)', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # RMSE
        axes[1].barh(model_names, rmse_scores, color=colors)
        axes[1].set_xlabel('RMSE')
        axes[1].set_title('RMSE Comparison (Lower is Better)', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # MAE
        axes[2].barh(model_names, mae_scores, color=colors)
        axes[2].set_xlabel('MAE')
        axes[2].set_title('MAE Comparison (Lower is Better)', fontsize=12)
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_predictions(y_true, y_pred, model_name='Model',
                        save_path='visualizations/predictions.png'):
        """
        Plot actual vs predicted values
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5, color='steelblue', s=50)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual WQI', fontsize=12)
        ax1.set_ylabel('Predicted WQI', fontsize=12)
        ax1.set_title(f'{model_name}: Actual vs Predicted', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Add text box with metrics
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Residual plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, color='coral', s=50)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted WQI', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title(f'{model_name}: Residual Plot', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction plot to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_feature_importance(model, feature_names, model_name='Model',
                               save_path='visualizations/feature_importance.png'):
        """
        Plot feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            
            plt.barh(range(len(feature_names)), importances[indices], color=colors)
            plt.yticks(range(len(feature_names)),
                      [feature_names[i].replace('_', ' ').title() for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name}: Feature Importance', fontsize=14)
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
            plt.close()
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute")

def create_all_visualizations():
    """
    Create all visualizations for the project
    """
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("Creating visualizations...")
    
    # Load data
    df = pd.read_csv('data/water_quality_train.csv')
    
    visualizer = WaterQualityVisualizer()
    
    # Create visualizations
    visualizer.plot_data_distribution(df)
    visualizer.plot_correlation_heatmap(df)
    visualizer.plot_wqi_by_category(df)
    
    # Model comparison (if models are trained)
    if os.path.exists('models/model_metrics.json'):
        visualizer.plot_model_comparison()
    
    print("\nAll visualizations created successfully!")

if __name__ == "__main__":
    create_all_visualizations()
