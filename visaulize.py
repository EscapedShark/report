import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns


# 7. Enhanced visualization
def visualize_results(y_true, y_pred, title="Prediction Results", result_dir=None, filename_prefix=""):
    """
    Enhanced prediction results visualization
    
    Parameters:
    y_true: True values
    y_pred: Predicted values
    title: Chart title
    result_dir: Directory to save results
    filename_prefix: Prefix for saved file names
    """
    plt.figure(figsize=(15, 10))
    
    # Main plot: Predicted vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.3)
    
    # Add ideal prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"{title} - Predicted vs Actual")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    
    # Add evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'RMSE: {rmse:.6f}\nR²: {r2:.6f}',
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.grid(True)
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals Analysis")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    
    # Residuals Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.grid(True)
    
    # Predicted vs Actual Distribution
    plt.subplot(2, 2, 4)
    sns.kdeplot(y_true, label="Actual")
    sns.kdeplot(y_pred, label="Predicted")
    plt.title("Distribution Comparison")
    plt.xlabel("Return Rate")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure if result_dir is provided
    if result_dir:
        plt.savefig(f"{result_dir}/figures/{filename_prefix}prediction_results.png", dpi=300)
    
    plt.close()  # Close figure to avoid display

def visualize_feature_importance(model, feature_names, top_n=20, result_dir=None, filename_prefix=""):
    """
    Visualize feature importance
    
    Parameters:
    model: Trained model
    feature_names: Feature names
    top_n: Show top N important features
    result_dir: Directory to save results
    filename_prefix: Prefix for saved file names
    """
    importance = model.feature_importance(importance_type='gain')
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names[:len(importance)],
        'Importance': importance
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Choose top N features
    top_features = feature_importance.head(top_n)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    # Save figure if result_dir is provided
    if result_dir:
        plt.savefig(f"{result_dir}/figures/{filename_prefix}feature_importance.png", dpi=300)
        
        # Save feature importance to CSV
        feature_importance.to_csv(f"{result_dir}/logs/{filename_prefix}feature_importance.csv", index=False)
    
    plt.close()  # Close figure to avoid display