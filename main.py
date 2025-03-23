import numpy as np
import pandas as pd
import sys
import datetime
import json
from datetime import datetime

from utilities import create_results_directory, Logger
from process_data import generate_stock_data, preprocess_data
from visaulize import visualize_results, visualize_feature_importance
from model import backtest_model, train_model, evaluate_model, save_model

# Set random seed for reproducibility
# 设置随机种子以确保结果可重复
np.random.seed(42)



# Main function
def main(use_time_series_split=False, run_backtest=False):
    """
    Main function
    
    Parameters:
    use_time_series_split: Whether to use time series split
    run_backtest: Whether to run backtesting
    
    Returns:
    model, test_metrics, result_dir
    """
    # Create results directory
    result_dir = create_results_directory()
    
    # Set up logging to both console and file
    log_file = open(f"{result_dir}/logs/execution_log.txt", 'w')
    sys.stdout = Logger(log_file)
    
    print("Starting stock return prediction model training...")
    print(f"Results will be saved to: {result_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: use_time_series_split={use_time_series_split}, run_backtest={run_backtest}")
    
    # Generate simulated data
    print("\nGenerating stock data...")
    stock_data = generate_stock_data(add_market_factor=True)
    print(f"Data generated, total rows: {len(stock_data)}")
    
    # Save sample of data
    stock_data.head(1000).to_csv(f"{result_dir}/logs/sample_data.csv", index=False)
    
    # Data preprocessing
    print("\nData preprocessing...")
    X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names, industry_dummies = preprocess_data(
        stock_data, time_series_split=use_time_series_split)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_valid.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save feature names
    pd.DataFrame({'feature': feature_names}).to_csv(f"{result_dir}/logs/feature_names.csv", index=False)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_valid, y_valid, result_dir=result_dir)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_train_pred, train_metrics = evaluate_model(model, X_train, y_train, feature_names, "Training", result_dir)
    y_valid_pred, valid_metrics = evaluate_model(model, X_valid, y_valid, feature_names, "Validation", result_dir)
    y_test_pred, test_metrics = evaluate_model(model, X_test, y_test, feature_names, "Test", result_dir)
    
    # Visualize feature importance
    print("\nVisualizing feature importance...")
    visualize_feature_importance(model, feature_names, 20, result_dir)
    
    # Visualize prediction results
    print("\nVisualizing prediction results...")
    visualize_results(y_test, y_test_pred, "Test Set", result_dir)
    
    # Save all metrics
    all_metrics = {
        'train': train_metrics,
        'validation': valid_metrics,
        'test': test_metrics
    }
    
    # Save model
    print("\nSaving model...")
    save_model(
        model, 
        f"{result_dir}/models", 
        feature_names, 
        None,  # No scaler to save
        all_metrics, 
        model.params
    )
    
    # Time series backtesting (if enabled)
    backtest_results = None
    if run_backtest:
        backtest_results = backtest_model(stock_data, result_dir=result_dir)
    
    # Model summary
    print("\nModel Summary:")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_valid)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test R²: {test_metrics['r2']:.6f}")
    print(f"Test Direction Accuracy: {test_metrics['direction_accuracy']:.4f}")
    
    # Save summary information
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'use_time_series_split': use_time_series_split,
            'run_backtest': run_backtest
        },
        'dataset_info': {
            'total_rows': len(stock_data),
            'train_samples': len(y_train),
            'valid_samples': len(y_valid),
            'test_samples': len(y_test),
            'feature_count': X_train.shape[1]
        },
        'performance': {
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics
        }
    }
    
    if backtest_results:
        summary['backtest'] = {
            'window_count': len(backtest_results),
            'avg_rmse': np.mean([r['metrics']['rmse'] for r in backtest_results]),
            'avg_r2': np.mean([r['metrics']['r2'] for r in backtest_results]),
            'avg_direction_accuracy': np.mean([r['metrics']['direction_accuracy'] for r in backtest_results])
        }
    
    with open(f"{result_dir}/logs/summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save important metrics
    sys.stdout.save_important_metrics(f"{result_dir}/logs/important_metrics.txt")
    
    # Close log file
    log_file.close()
    sys.stdout = sys.__stdout__  # Restore original stdout
    
    print(f"All results have been saved to: {result_dir}")
    
    return model, test_metrics, result_dir

if __name__ == "__main__":
    # Run main function (can adjust parameters as needed)
    model, metrics, result_dir = main(use_time_series_split=True, run_backtest=False)