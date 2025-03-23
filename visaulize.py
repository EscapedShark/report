import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


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

def plot_learning_curve(eval_hist, result_dir):
    """
    绘制学习曲线并保存历史数据
    
    参数:
    eval_hist: 包含训练历史的字典
    result_dir: 保存结果的目录
    """
    if not eval_hist or 'iteration' not in eval_hist or len(eval_hist['iteration']) == 0:
        print("警告: 没有足够的训练历史数据来绘制学习曲线")
        return
        
    try:
        # 绘制学习曲线
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
        
        # 保存历史数据为CSV
        hist_df = pd.DataFrame(eval_hist)
        hist_df.to_csv(f"{result_dir}/logs/training_history.csv", index=False)
    except Exception as e:
        print(f"警告: 无法绘制学习曲线 - {str(e)}")
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

def plot_backtest_results(backtest_results, result_dir=None):
    """
    绘制回测结果图表
    
    参数:
    backtest_results: 回测结果列表
    result_dir: 保存结果的目录
    """
    # 创建性能可视化图表
    plt.figure(figsize=(15, 10))
    
    # 绘制RMSE子图
    plt.subplot(2, 2, 1)
    plt.title("Backtest RMSE by Window")
    plt.xlabel("Window")
    plt.ylabel("RMSE")
    plt.grid(True)
    
    # 绘制R²子图
    plt.subplot(2, 2, 2)
    plt.title("Backtest R² by Window")
    plt.xlabel("Window")
    plt.ylabel("R²")
    plt.grid(True)
    
    # 绘制方向准确率子图
    plt.subplot(2, 2, 3)
    plt.title("Backtest Direction Accuracy by Window")
    plt.xlabel("Window")
    plt.ylabel("Direction Accuracy")
    plt.grid(True)
    
    # 如果有回测结果，绘制数据点
    if backtest_results:
        window_nums = [result['window'] for result in backtest_results]
        rmse_values = [result['metrics']['rmse'] for result in backtest_results]
        r2_values = [result['metrics']['r2'] for result in backtest_results]
        direction_accuracy_values = [result['metrics']['direction_accuracy'] for result in backtest_results]
        
        # 绘制每个窗口的指标
        plt.subplot(2, 2, 1)
        plt.bar(window_nums, rmse_values)
        
        plt.subplot(2, 2, 2)
        plt.bar(window_nums, r2_values)
        
        plt.subplot(2, 2, 3)
        plt.bar(window_nums, direction_accuracy_values)
        
        # 计算平均指标
        avg_rmse = np.mean(rmse_values)
        avg_r2 = np.mean(r2_values)
        avg_dir_acc = np.mean(direction_accuracy_values)
        
        # 绘制摘要子图
        plt.subplot(2, 2, 4)
        plt.title("Backtest Performance Summary")
        plt.axis('off')
        
        summary_text = (
            f"Average RMSE: {avg_rmse:.6f}\n"
            f"Average R²: {avg_r2:.6f}\n"
            f"Average Direction Accuracy: {avg_dir_acc:.4f}\n"
            f"Number of windows: {len(backtest_results)}"
        )
        
        plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        
        # 打印回测摘要
        print("\nBacktest Summary:")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average R²: {avg_r2:.6f}")
        print(f"Average Direction Accuracy: {avg_dir_acc:.4f}")
    
    plt.tight_layout()
    
    # 保存回测摘要图
    if result_dir:
        plt.savefig(f"{result_dir}/figures/backtest_summary.png", dpi=300)
    
    plt.close()  # 关闭图形以避免显示