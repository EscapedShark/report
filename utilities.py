
from datetime import datetime
import os
import sys

# 模型文件夹创建函数
def create_results_directory(base_dir="stock_model_results"):
    """
    Create a directory structure for storing results
    
    Parameters:
    base_dir: Base directory name
    
    Returns:
    Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{base_dir}_{timestamp}"
    
    # Create main directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{result_dir}/figures", exist_ok=True)
    os.makedirs(f"{result_dir}/logs", exist_ok=True)
    os.makedirs(f"{result_dir}/models", exist_ok=True)
    
    return result_dir

# 回测文件夹创建函数
def ensure_directory_structure(directory):
    """
    Ensure that a directory and its necessary subdirectories exist
    
    Parameters:
    directory: Base directory path
    
    Returns:
    None
    """
    if directory is None:
        return
    
    # Create main directory
    os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{directory}/figures", exist_ok=True)
    os.makedirs(f"{directory}/logs", exist_ok=True)


class Logger:
    """Custom logger to write to both console and file"""
    
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.log_texts = []  # Store important log texts
        
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
        
        # Store non-empty messages
        if message.strip():
            self.log_texts.append(message.strip())
        
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def save_important_metrics(self, file_path):
        """Save important metrics to a separate file"""
        # Filter for important metrics
        metrics = []
        for text in self.log_texts:
            if any(keyword in text for keyword in ["RMSE:", "R²:", "Direction Accuracy:", "Feature count:", "Average"]):
                metrics.append(text)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write("=== IMPORTANT METRICS ===\n\n")
            for metric in metrics:
                f.write(f"{metric}\n")