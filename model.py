import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import datetime
import json
from datetime import datetime

from utilities import create_results_directory, ensure_directory_structure, Logger
from process_data import generate_stock_data, preprocess_data
from visaulize import visualize_results, visualize_feature_importance

# Set random seed for reproducibility
np.random.seed(42)

# Model evaluation
def evaluate_model(model, X, y, feature_names, dataset_name="", result_dir=None):
    """
    Enhanced model evaluation
    
    Parameters:
    model: Trained model
    X, y: Evaluation data
    feature_names: Feature name list
    dataset_name: Dataset name
    result_dir: Directory to save results
    
    Returns:
    y_pred, metrics_dict
    """
    y_pred = model.predict(X)
    
    # Calculate multiple evaluation metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Avoid divide by zero warning
    non_zero_mask = y != 0
    y_non_zero = y[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    mape = np.mean(np.abs((y_non_zero - y_pred_non_zero) / y_non_zero)) * 100 if len(y_non_zero) > 0 else float('inf')
    
    # Calculate direction accuracy (whether predicted return sign is correct)
    direction_accuracy = np.mean((y > 0) == (y_pred > 0))
    
    print(f"\n{dataset_name} Set Evaluation Results:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
    
    # Feature importance analysis
    if dataset_name == "Test":
        feature_importance = model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importance)],
            'Importance': feature_importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        # Save feature importance to file if result_dir is provided
        if result_dir:
            importance_df.to_csv(f"{result_dir}/logs/{dataset_name.lower()}_feature_importance.csv", index=False)
    
    # Save metrics to file if result_dir is provided
    if result_dir:
        with open(f"{result_dir}/logs/{dataset_name.lower()}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
    
    return y_pred, metrics

# Model saving
def save_model(model, model_dir, feature_names=None, scaler=None, all_metrics=None, params=None):
    """
    Save model and related components
    
    Parameters:
    model: Trained model
    model_dir: Save path
    feature_names: Feature names
    scaler: Data standardizer
    all_metrics: Dictionary of metrics for all datasets
    params: Model parameters
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model.save_model(f"{model_dir}/lgbm_model.txt")
    
    # Save feature names
    if feature_names is not None:
        np.save(f"{model_dir}/feature_names.npy", feature_names)
    
    # Save standardizer
    if scaler is not None:
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    
    # Save metrics
    if all_metrics is not None:
        with open(f"{model_dir}/model_metrics.json", 'w') as f:
            json.dump(all_metrics, f, indent=4)
    
    # Save parameters
    if params is not None:
        with open(f"{model_dir}/model_params.json", 'w') as f:
            json.dump(params, f, indent=4)
    
    print(f"\nModel and components saved to {model_dir}")

# Model loading
def load_model(model_path):
    """
    Load model and related components
    
    Parameters:
    model_path: Model save path
    
    Returns:
    model, feature_names, scaler, metrics, params
    """
    # Load model
    model = lgb.Booster(model_file=f"{model_path}/lgbm_model.txt")
    
    # Load feature names
    feature_names = None
    if os.path.exists(f"{model_path}/feature_names.npy"):
        feature_names = np.load(f"{model_path}/feature_names.npy", allow_pickle=True)
    
    # Load standardizer
    scaler = None
    if os.path.exists(f"{model_path}/scaler.pkl"):
        scaler = joblib.load(f"{model_path}/scaler.pkl")
    
    # Load metrics
    metrics = None
    if os.path.exists(f"{model_path}/model_metrics.json"):
        with open(f"{model_path}/model_metrics.json", 'r') as f:
            metrics = json.load(f)
    
    # Load parameters
    params = None
    if os.path.exists(f"{model_path}/model_params.json"):
        with open(f"{model_path}/model_params.json", 'r') as f:
            params = json.load(f)
    
    print(f"\nLoaded model and components from {model_path}")
    
    return model, feature_names, scaler, metrics, params

# Model training
def train_model(X_train, y_train, X_valid, y_valid, params=None, result_dir=None):
    """
    Train LightGBM model
    
    Parameters:
    X_train, y_train: Training data
    X_valid, y_valid: Validation data
    params: Model parameters (if None, use default parameters)
    result_dir: Directory to save results
    
    Returns:
    Trained model
    """
    # Ensure directory structure exists if result_dir is provided
    if result_dir:
        ensure_directory_structure(result_dir)
    
    # Prepare LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # Use default parameters if none provided
    if params is None:
        # Default parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'early_stopping_round': 50,
            'verbose': -1 #10
        }
    
    # Save parameters to file if result_dir is provided
    if result_dir:
        with open(f"{result_dir}/logs/model_parameters.json", 'w') as f:
            json.dump(params, f, indent=4)
    
    # 创建变量保存训练历史
    eval_hist = {}
    
    # 回调函数收集训练历史
    def eval_callback(env):
        if 'iteration' not in eval_hist:
            eval_hist['iteration'] = []
            eval_hist['train_rmse'] = []
            eval_hist['valid_rmse'] = []
        
        iteration = env.iteration
        eval_hist['iteration'].append(iteration)
        
        # 确保有足够的评估结果
        if len(env.evaluation_result_list) >= 2:
            train_metric = env.evaluation_result_list[0][2]
            valid_metric = env.evaluation_result_list[1][2]
            eval_hist['train_rmse'].append(train_metric)
            eval_hist['valid_rmse'].append(valid_metric)
    
    # 训练模型
    try:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            valid_names=['training', 'valid_0'],
            callbacks=[eval_callback]
        )
    except Exception as e:
        # 如果回调方法失败，回退到不使用回调的训练
        print(f"Warning: Training with callback failed. Falling back to basic training: {str(e)}")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data]
        )
    
    # Plot learning curve if result_dir is provided
    if result_dir and eval_hist and 'iteration' in eval_hist and len(eval_hist['iteration']) > 0:
        try:
            # 使用我们收集的历史数据绘制学习曲线
            plt.figure(figsize=(10, 6))
            plt.plot(eval_hist['iteration'], eval_hist['train_rmse'], label='Training RMSE')
            plt.plot(eval_hist['iteration'], eval_hist['valid_rmse'], label='Validation RMSE')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Learning Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{result_dir}/figures/learning_curve.png", dpi=300)
            plt.close()
            
            # 同时保存历史数据为CSV
            hist_df = pd.DataFrame(eval_hist)
            hist_df.to_csv(f"{result_dir}/logs/training_history.csv", index=False)
        except Exception as e:
            print(f"Warning: Could not plot learning curve - {str(e)}")
            try:
                # 创建替代图像
                plt.figure(figsize=(10, 6))
                plt.title("Model Training Process")
                plt.text(0.5, 0.5, "Model trained successfully but learning curve visualization failed", 
                        ha='center', va='center', fontsize=12)
                plt.savefig(f"{result_dir}/figures/learning_curve_alternative.png", dpi=300)
                plt.close()
            except:
                pass
    
    return model

# Backtesting function
def backtest_model(df, window_size=60, step=20, test_size=20, result_dir=None):
    """
    Time series backtesting with consistent feature dimensions
    
    Parameters:
    df: Stock data
    window_size: Training window size
    step: Step size
    test_size: Test set size
    result_dir: Directory to save results
    
    Returns:
    Backtest results
    """
    print("\nStarting time series backtesting...")
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get unique dates
    dates = df['date'].unique()
    
    if len(dates) < window_size + test_size:
        print("Insufficient data for backtesting")
        return None
    
    # Create consistent industry dummies for entire dataset
    all_industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
    
    # Preserve the full feature list for consistent predictions
    basic_features = [f'feature_{i+1}' for i in range(20)]
    price_features = ['price', 'price_change', 'return', 'volatility']
    tech_features = ['MA5', 'MA10', 'MA20', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI', 
                    'BB_middle', 'BB_upper', 'BB_lower', 'price_to_MA5', 'price_to_MA20', 'MA5_cross_MA20']
    lag_features = [f'price_lag_{i}' for i in range(1, 6)] + [f'return_lag_{i}' for i in range(1, 6)]
    
    # If data contains market factor, add to features
    if 'market_factor' in df.columns:
        market_features = ['market_factor']
    else:
        market_features = []
    
    # Combine all features
    all_features = basic_features + price_features + tech_features + lag_features + market_features
    
    # Get complete feature names (including all industry dummies)
    complete_feature_names = all_features + all_industry_dummies.columns.tolist()
    
    # Initialize results
    backtest_results = []
    
    # Create a figure for performance visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("Backtest RMSE by Window")
    plt.xlabel("Window")
    plt.ylabel("RMSE")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.title("Backtest R² by Window")
    plt.xlabel("Window")
    plt.ylabel("R²")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.title("Backtest Direction Accuracy by Window")
    plt.xlabel("Window")
    plt.ylabel("Direction Accuracy")
    plt.grid(True)
    
    # Perform backtesting
    for i in range(0, len(dates) - window_size - test_size + 1, step):
        # Determine training and testing dates
        train_dates = dates[i:i+window_size]
        test_dates = dates[i+window_size:i+window_size+test_size]
        
        window_num = i//step + 1
        print(f"\nBacktest Window {window_num}:")
        print(f"Training period: {train_dates[0]} to {train_dates[-1]}")
        print(f"Testing period: {test_dates[0]} to {test_dates[-1]}")
        
        # Split data
        train_data = df[df['date'].isin(train_dates)]
        test_data = df[df['date'].isin(test_dates)]
        
        # Create subdirectory for this window if result_dir is provided
        window_dir = None
        if result_dir:
            window_dir = f"{result_dir}/backtest_window_{window_num}"
            ensure_directory_structure(window_dir)
            
        try:
            # Process training data with full industry dummies set
            X_train_features = train_data[all_features].values
            y_train = train_data['future_return'].values
            
            # Standardize features
            scaler = StandardScaler()
            X_train_features = scaler.fit_transform(X_train_features)
            
            # Create consistent industry dummies for train data
            train_dummies = pd.DataFrame(0, index=train_data.index, columns=all_industry_dummies.columns)
            temp_train_dummies = pd.get_dummies(train_data['industry'], prefix='industry')
            for col in temp_train_dummies.columns:
                if col in train_dummies.columns:
                    train_dummies[col] = temp_train_dummies[col]
            
            # Combine features and dummies
            X_train = np.hstack([X_train_features, train_dummies.values])
            
            # Split into train and validation
            split_idx = int(len(X_train) * 0.8)
            X_train_subset, X_valid = X_train[:split_idx], X_train[split_idx:]
            y_train_subset, y_valid = y_train[:split_idx], y_train[split_idx:]
            
            # Process test data with same feature set
            X_test_features = test_data[all_features].values
            y_test = test_data['future_return'].values
            
            # Standardize using same scaler
            X_test_features = scaler.transform(X_test_features)
            
            # Create consistent industry dummies for test data
            test_dummies = pd.DataFrame(0, index=test_data.index, columns=all_industry_dummies.columns)
            temp_test_dummies = pd.get_dummies(test_data['industry'], prefix='industry')
            for col in temp_test_dummies.columns:
                if col in test_dummies.columns:
                    test_dummies[col] = temp_test_dummies[col]
            
            # Combine features and dummies
            X_test = np.hstack([X_test_features, test_dummies.values])
            
            # Verify feature dimensions are the same
            if X_train.shape[1] != X_test.shape[1]:
                print(f"Warning: Feature dimension mismatch - Training: {X_train.shape[1]}, Testing: {X_test.shape[1]}")
                print("Adjusting feature dimensions to match...")
                
                # This shouldn't happen with our approach, but add this as a failsafe
                min_dim = min(X_train.shape[1], X_test.shape[1])
                X_train = X_train[:, :min_dim]
                X_test = X_test[:, :min_dim]
                X_train_subset = X_train_subset[:, :min_dim]
                X_valid = X_valid[:, :min_dim]
                
                print(f"Adjusted to common feature dimension: {min_dim}")
            
            # Debug information
            print(f"Feature dimensions - Training: {X_train.shape[1]}, Testing: {X_test.shape[1]}")
            
            # Train model
            model = train_model(X_train_subset, y_train_subset, X_valid, y_valid, result_dir=window_dir)
            
            # Evaluate model - Use predict_disable_shape_check=True as a safety measure
            y_pred = model.predict(X_test, predict_disable_shape_check=True)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))
            
            metrics = {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
            
            print(f"\nBacktest Window {window_num} Evaluation Results:")
            print(f"RMSE: {rmse:.6f}")
            print(f"R²: {r2:.6f}")
            print(f"Direction Accuracy: {direction_accuracy:.4f}")
            
            # Save window results if result_dir is provided
            if result_dir and window_dir:
                # Save predictions
                pred_df = pd.DataFrame({
                    'actual': y_test,
                    'predicted': y_pred
                })
                pred_df.to_csv(f"{window_dir}/logs/predictions.csv", index=False)
                
                # Save metrics
                with open(f"{window_dir}/logs/metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Create visualization
                visualize_results(y_test, y_pred, f"Window {window_num}", window_dir)
                
                # Use only the available feature names that match the model
                actual_feature_names = complete_feature_names[:X_train.shape[1]]
                visualize_feature_importance(model, actual_feature_names, 15, window_dir)
            
            # Add to plot
            plt.subplot(2, 2, 1)
            plt.bar(window_num, rmse)
            
            plt.subplot(2, 2, 2)
            plt.bar(window_num, r2)
            
            plt.subplot(2, 2, 3)
            plt.bar(window_num, direction_accuracy)
            
            # Save results
            backtest_results.append({
                'window': window_num,
                'train_start': train_dates[0].strftime('%Y-%m-%d') if hasattr(train_dates[0], 'strftime') else str(train_dates[0]),
                'train_end': train_dates[-1].strftime('%Y-%m-%d') if hasattr(train_dates[-1], 'strftime') else str(train_dates[-1]),
                'test_start': test_dates[0].strftime('%Y-%m-%d') if hasattr(test_dates[0], 'strftime') else str(test_dates[0]),
                'test_end': test_dates[-1].strftime('%Y-%m-%d') if hasattr(test_dates[-1], 'strftime') else str(test_dates[-1]),
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"Error processing window {window_num}: {str(e)}")
            print("Skipping this window and continuing...")
            
            # Add more detailed error debugging
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            
            # Save error information if result_dir is provided
            if result_dir and window_dir:
                with open(f"{window_dir}/logs/error.txt", 'w') as f:
                    f.write(f"Error processing window {window_num}: {str(e)}\n\n")
                    traceback.print_exc(file=f)
    
    # Complete backtest summary plot
    plt.subplot(2, 2, 4)
    plt.title("Backtest Performance Summary")
    plt.axis('off')
    
    # Add backtest summary to plot if there are results
    if backtest_results:
        rmse_values = [result['metrics']['rmse'] for result in backtest_results]
        r2_values = [result['metrics']['r2'] for result in backtest_results]
        direction_accuracy_values = [result['metrics']['direction_accuracy'] for result in backtest_results]
        
        avg_rmse = np.mean(rmse_values)
        avg_r2 = np.mean(r2_values)
        avg_dir_acc = np.mean(direction_accuracy_values)
        
        summary_text = (
            f"Average RMSE: {avg_rmse:.6f}\n"
            f"Average R²: {avg_r2:.6f}\n"
            f"Average Direction Accuracy: {avg_dir_acc:.4f}\n"
            f"Number of windows: {len(backtest_results)}"
        )
        
        plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        
        # Analyze backtest results
        print("\nBacktest Summary:")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average R²: {avg_r2:.6f}")
        print(f"Average Direction Accuracy: {avg_dir_acc:.4f}")
    
    plt.tight_layout()
    
    # Save backtest summary if result_dir is provided
    if result_dir:
        plt.savefig(f"{result_dir}/figures/backtest_summary.png", dpi=300)
        
        # Save detailed backtest results
        if backtest_results:
            with open(f"{result_dir}/logs/backtest_results.json", 'w') as f:
                json.dump(backtest_results, f, indent=4)
                
            # Create DataFrame for easy analysis
            metrics_list = []
            for result in backtest_results:
                window_metrics = {
                    'window': result['window'],
                    'train_start': result['train_start'],
                    'train_end': result['train_end'],
                    'test_start': result['test_start'],
                    'test_end': result['test_end'],
                }
                window_metrics.update(result['metrics'])
                metrics_list.append(window_metrics)
            
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(f"{result_dir}/logs/backtest_metrics.csv", index=False)
    
    plt.close()  # Close figure to avoid display
    
    return backtest_results

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