import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json


from utilities import ensure_directory_structure
from visaulize import visualize_results, visualize_feature_importance, plot_learning_curve, plot_backtest_results

# Model evaluation 模型评估
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

# Model saving 模型保存
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

# Model training 模型训练
def train_model(X_train, y_train, X_valid, y_valid, params=None, result_dir=None):
    """
    训练LightGBM模型
    
    参数:
    X_train, y_train: 训练数据
    X_valid, y_valid: 验证数据
    params: 模型参数（如果为None，使用默认参数）
    result_dir: 保存结果的目录
    
    返回:
    训练好的模型
    """
    # 确保目录结构存在
    if result_dir:
        ensure_directory_structure(result_dir)
    
    # 准备LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 使用默认参数
    if params is None:
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
            'verbose': -1  # 避免打印过多信息
        }
    
    # 保存参数到文件
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
    
    model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            valid_names=['training', 'valid_0'],
            callbacks=[eval_callback]
        )

    

    if result_dir:
        plot_learning_curve(eval_hist, result_dir)
    
    return model

# backtesting 回测
def backtest_model(df, window_size=60, step=20, test_size=20, result_dir=None):
    """
    时间序列回测，具有一致的特征维度
    
    参数:
    df: 股票数据
    window_size: 训练窗口大小
    step: 步长
    test_size: 测试集大小
    result_dir: 保存结果的目录
    
    返回:
    回测结果
    """
    print("\nStarting time series backtesting...")
    
    # 按日期排序
    df = df.sort_values('date')
    
    # 获取唯一日期
    dates = df['date'].unique()
    
    if len(dates) < window_size + test_size:
        print("Insufficient data for backtesting")
        return None
    
    # 为整个数据集创建一致的行业虚拟变量
    all_industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
    
    # 保留完整特征列表
    basic_features = [f'feature_{i+1}' for i in range(20)]
    price_features = ['price', 'price_change', 'return', 'volatility']
    tech_features = ['MA5', 'MA10', 'MA20', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI', 
                    'BB_middle', 'BB_upper', 'BB_lower', 'price_to_MA5', 'price_to_MA20', 'MA5_cross_MA20']
    lag_features = [f'price_lag_{i}' for i in range(1, 6)] + [f'return_lag_{i}' for i in range(1, 6)]
    
    # 如果数据包含市场因子，则添加到特征
    if 'market_factor' in df.columns:
        market_features = ['market_factor']
    else:
        market_features = []
    
    # 合并所有特征
    all_features = basic_features + price_features + tech_features + lag_features + market_features
    
    # 获取完整特征名称（包括所有行业虚拟变量）
    complete_feature_names = all_features + all_industry_dummies.columns.tolist()
    
    # 初始化结果
    backtest_results = []
    
    # 执行回测
    for i in range(0, len(dates) - window_size - test_size + 1, step):
        # 确定训练和测试日期
        train_dates = dates[i:i+window_size]
        test_dates = dates[i+window_size:i+window_size+test_size]
        
        window_num = i//step + 1
        print(f"\nBacktest Window {window_num}:")
        print(f"Training period: {train_dates[0]} to {train_dates[-1]}")
        print(f"Testing period: {test_dates[0]} to {test_dates[-1]}")
        
        # 分割数据
        train_data = df[df['date'].isin(train_dates)]
        test_data = df[df['date'].isin(test_dates)]
        
        # 为此窗口创建子目录
        window_dir = None
        if result_dir:
            window_dir = f"{result_dir}/backtest_window_{window_num}"
            ensure_directory_structure(window_dir)
            
        try:
            # 处理训练数据
            X_train_features = train_data[all_features].values
            y_train = train_data['future_return'].values
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_features = scaler.fit_transform(X_train_features)
            
            # 为训练数据创建一致的行业虚拟变量
            train_dummies = pd.DataFrame(0, index=train_data.index, columns=all_industry_dummies.columns)
            temp_train_dummies = pd.get_dummies(train_data['industry'], prefix='industry')
            for col in temp_train_dummies.columns:
                if col in train_dummies.columns:
                    train_dummies[col] = temp_train_dummies[col]
            
            # 合并特征和虚拟变量
            X_train = np.hstack([X_train_features, train_dummies.values])
            
            # 分割为训练集和验证集
            split_idx = int(len(X_train) * 0.8)
            X_train_subset, X_valid = X_train[:split_idx], X_train[split_idx:]
            y_train_subset, y_valid = y_train[:split_idx], y_train[split_idx:]
            
            # 处理测试数据
            X_test_features = test_data[all_features].values
            y_test = test_data['future_return'].values
            
            # 使用相同的标准化器
            X_test_features = scaler.transform(X_test_features)
            
            # 为测试数据创建一致的行业虚拟变量
            test_dummies = pd.DataFrame(0, index=test_data.index, columns=all_industry_dummies.columns)
            temp_test_dummies = pd.get_dummies(test_data['industry'], prefix='industry')
            for col in temp_test_dummies.columns:
                if col in test_dummies.columns:
                    test_dummies[col] = temp_test_dummies[col]
            
            # 合并特征和虚拟变量
            X_test = np.hstack([X_test_features, test_dummies.values])
            
            # 验证特征维度是否相同
            if X_train.shape[1] != X_test.shape[1]:
                print(f"Warning: Feature dimension mismatch - Training: {X_train.shape[1]}, Testing: {X_test.shape[1]}")
                print("Adjusting feature dimensions to match...")
                
                # 调整为共同维度
                min_dim = min(X_train.shape[1], X_test.shape[1])
                X_train = X_train[:, :min_dim]
                X_test = X_test[:, :min_dim]
                X_train_subset = X_train_subset[:, :min_dim]
                X_valid = X_valid[:, :min_dim]
                
                print(f"Adjusted to common feature dimension: {min_dim}")
            
            # 调试信息
            print(f"Feature dimensions - Training: {X_train.shape[1]}, Testing: {X_test.shape[1]}")
            
            # 训练模型
            model = train_model(X_train_subset, y_train_subset, X_valid, y_valid, result_dir=window_dir)
            
            # 评估模型 - 使用predict_disable_shape_check=True作为安全措施
            y_pred = model.predict(X_test, predict_disable_shape_check=True)
            
            # 计算指标
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
            
            # 保存窗口结果
            if result_dir and window_dir:
                # 保存预测
                pred_df = pd.DataFrame({
                    'actual': y_test,
                    'predicted': y_pred
                })
                pred_df.to_csv(f"{window_dir}/logs/predictions.csv", index=False)
                
                # 保存指标
                with open(f"{window_dir}/logs/metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # 创建可视化
                visualize_results(y_test, y_pred, f"Window {window_num}", window_dir)
                
                # 使用匹配模型的可用特征名称
                actual_feature_names = complete_feature_names[:X_train.shape[1]]
                visualize_feature_importance(model, actual_feature_names, 15, window_dir)
            
            # 保存结果
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
            
            # 添加更详细的错误调试
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            
            # 保存错误信息
            if result_dir and window_dir:
                with open(f"{window_dir}/logs/error.txt", 'w') as f:
                    f.write(f"Error processing window {window_num}: {str(e)}\n\n")
                    traceback.print_exc(file=f)
    
    # 绘制回测结果
    plot_backtest_results(backtest_results, result_dir)
    
    # 保存详细回测结果
    if result_dir and backtest_results:
        with open(f"{result_dir}/logs/backtest_results.json", 'w') as f:
            json.dump(backtest_results, f, indent=4)
            
        # 创建DataFrame用于简单分析
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
    
    return backtest_results