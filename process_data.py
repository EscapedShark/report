import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def generate_stock_data(n_stocks=100, n_days=500, n_industries=10, add_market_factor=True):
    """
    生成增强的模拟股票数据，特征重要性更加平衡
    
    Parameters:
    n_stocks: 股票数量
    n_days: 天数
    n_industries: 行业数量
    add_market_factor: 是否添加市场因子
    
    Returns:
    包含股票数据的pandas DataFrame
    """
    # 创建时间索引
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    # 创建股票代码和行业分类
    stocks = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    industry_weights = np.random.dirichlet(np.ones(n_industries) * 2, 1)[0]
    industries = np.random.choice(np.arange(n_industries), n_stocks, p=industry_weights)
    
    # 创建行业特性 - 更复杂的模型
    industry_base_effects = np.random.normal(0, 0.05, n_industries)  # 基础效应
    industry_cyclicality = np.random.uniform(0.5, 2.0, n_industries)  # 周期性强度
    industry_trends = np.random.normal(0, 0.0003, n_industries)  # 长期趋势
    
    # 行业间相关矩阵
    industry_correlation = np.eye(n_industries)
    for i in range(n_industries):
        for j in range(i+1, n_industries):
            corr = np.random.uniform(-0.3, 0.7)
            industry_correlation[i, j] = corr
            industry_correlation[j, i] = corr
    
    # 生成市场因子
    market_factor = np.zeros(n_days)
    
    if add_market_factor:
        # 模拟不同的市场状态
        regimes = ['bull', 'bear', 'sideways']
        current_regime = np.random.choice(regimes, p=[0.5, 0.3, 0.2])
        regime_duration = int(np.random.gamma(shape=5, scale=10)) + 20
        
        # 市场趋势和波动参数
        trend_params = {
            'bull': (0.0006, 0.008),
            'bear': (-0.0007, 0.012),
            'sideways': (0.0001, 0.006)
        }
        
        # 生成市场因子
        market_factor[0] = np.random.normal(0, 0.01)
        day = 1
        
        while day < n_days:
            # 可能切换市场状态
            if day % regime_duration == 0:
                transition_probs = {
                    'bull': {'bull': 0.7, 'bear': 0.2, 'sideways': 0.1},
                    'bear': {'bull': 0.3, 'bear': 0.5, 'sideways': 0.2},
                    'sideways': {'bull': 0.4, 'bear': 0.3, 'sideways': 0.3}
                }
                
                next_regime_probs = transition_probs[current_regime]
                regime_options = list(next_regime_probs.keys())
                regime_probs = list(next_regime_probs.values())
                
                new_regime = np.random.choice(regime_options, p=regime_probs)
                current_regime = new_regime
                regime_duration = int(np.random.gamma(shape=5, scale=10)) + 20
            
            # 获取当前状态的趋势和波动参数
            trend, vol = trend_params[current_regime]
            
            # 应用随机冲击
            if np.random.random() < 0.005:
                shock = np.random.normal(0, 0.03) * (-1 if current_regime == 'bull' else 1)
                market_factor[day] = market_factor[day-1] + shock
            else:
                # 正常市场波动
                market_factor[day] = market_factor[day-1] + np.random.normal(trend, vol)
            
            day += 1
    
    # 初始化数据列表
    data_rows = []
    
    # 为每只股票生成独特的风险特性
    stock_betas = np.random.lognormal(mean=0, sigma=0.4, size=n_stocks)  # 市场敏感度
    stock_volatility_factor = np.random.gamma(2, 0.5, n_stocks)  # 波动性特性
    
    # 生成特征重要性权重 - 决定每个特征对未来收益率的影响
    # 为20个特征分配不同的影响权重，这些权重将用于计算未来收益率
    feature_importance_weights = np.random.exponential(1, 20)
    # 标准化，确保权重总和为1
    feature_importance_weights = feature_importance_weights / np.sum(feature_importance_weights)
    
    # 为特征间添加相关结构
    feature_correlation = np.eye(20)
    for i in range(20):
        for j in range(i+1, 20):
            # 添加一些相关性，但不要太强
            corr = np.random.uniform(-0.3, 0.3)
            feature_correlation[i, j] = corr
            feature_correlation[j, i] = corr
    
    # 创建特征组，用于非线性交互
    feature_groups = []
    for _ in range(5):  # 创建5个特征组
        group_size = np.random.randint(2, 5)  # 每组2-4个特征
        group = np.random.choice(20, group_size, replace=False)
        feature_groups.append(group)
    
    for stock_idx, stock in enumerate(stocks):
        industry = industries[stock_idx]
        beta = stock_betas[stock_idx]
        
        # 基础特征值
        base_features = np.random.normal(0, 1, 20)
        
        # 初始股票价格
        price = np.random.uniform(10, 100)
        prices = []
        returns = []  # 保存日收益率
        
        # 股票特有的特征权重修改
        # 每只股票对特征的反应程度不同
        stock_feature_weights = feature_importance_weights * (1 + np.random.normal(0, 0.3, 20))
        
        for day in range(n_days):
            # 特征随时间变化
            if day == 0:
                features = base_features + np.random.normal(0, 0.1, 20)
            else:
                # 特征有粘性，部分保留昨天的值
                prev_features = np.array([data_rows[-1][f'feature_{i+1}'] for i in range(20)])
                innovations = np.random.multivariate_normal(np.zeros(20), np.eye(20) * 0.05)
                
                # 应用特征相关性
                correlated_innovations = np.dot(feature_correlation, innovations)
                features = prev_features * 0.95 + correlated_innovations * 0.05
            
            # 计算当天价格和收益率
            if day > 0:
                # 基础收益率计算
                # 关键修改：使用特征和其他因素构建当天收益率，而不是直接使用过去收益率
                
                # 1. 基于特征的线性影响
                feature_linear_impact = np.sum(features * np.random.normal(0, 0.005, 20))
                
                # 2. 特征的非线性交互影响
                feature_nonlinear_impact = 0
                for group in feature_groups:
                    # 计算组内特征的交互作用
                    interaction = np.prod(features[group]) * np.random.normal(0, 0.001)
                    feature_nonlinear_impact += interaction
                
                # 3. 行业基础效应和趋势
                industry_effect = industry_base_effects[industry] + industry_trends[industry] * day
                
                # 4. 市场因子影响
                market_effect = 0
                if add_market_factor:
                    market_effect = beta * market_factor[day] * industry_cyclicality[industry]
                
                # 5. 随机噪声 - 使用t分布
                df = 5  # t分布自由度
                noise = np.random.standard_t(df=df) * 0.01 * stock_volatility_factor[stock_idx]
                
                # 计算当天收益率 - 不使用过去的收益率作为输入
                day_return = (
                    feature_linear_impact + 
                    feature_nonlinear_impact +
                    industry_effect + 
                    market_effect + 
                    noise
                )
                
                # 添加一些很小的随机跳跃
                if np.random.random() < 0.01:  # 1%概率
                    day_return += np.random.normal(0, 0.02)
                
                # 更新价格
                price = price * (1 + day_return)
                price = max(price, 0.01)  # 确保价格为正
                
                # 保存收益率
                returns.append(day_return)
            
            prices.append(price)
            
            # 计算未来5天的加权收益率
            # 关键修改：确保future_return不会太容易从当天return预测
            if day < n_days - 5:
                weights = np.linspace(0.5, 0.1, 5)  # 未来日期权重递减
                future_days_returns = []
                
                # 关键修改：这里改为为每个未来日计算独立的收益率
                for future_day in range(1, 6):
                    # 为未来每一天计算其独特的特征影响
                    # 创建特征的预期变化 - 特征在未来几天如何演变
                    future_features = features.copy()
                    for i in range(20):
                        # 特征随时间逐渐变化，变化程度与时间和波动性有关
                        future_features[i] += np.random.normal(0, 0.05 * future_day)
                    
                    # 基于未来特征计算收益率
                    future_feature_impact = np.sum(future_features * stock_feature_weights * np.random.normal(0.8, 0.2, 20))
                    
                    # 添加未来市场影响（如果可用）
                    future_market_impact = 0
                    if add_market_factor and day + future_day < n_days:
                        future_market_impact = beta * market_factor[day + future_day - 1] * industry_cyclicality[industry]
                    
                    # 添加行业效应
                    future_industry_effect = industry_base_effects[industry] + industry_trends[industry] * (day + future_day)
                    
                    # 添加大量噪声 - 确保未来收益率难以预测
                    future_noise_level = 0.01 * (future_day) * stock_volatility_factor[stock_idx]
                    future_noise = np.random.standard_t(df=5) * future_noise_level
                    
                    # 如果是第一天的未来收益率，添加与当前收益率的小相关性（但不是主要因素）
                    curr_return_effect = 0
                    if day > 0 and future_day == 1:
                        # 当前收益率对未来的影响，但不是主导因素
                        # 仅影响约10-20%
                        if len(returns) > 0:
                            curr_return_effect = returns[-1] * np.random.uniform(0.1, 0.2)
                    
                    # 计算这一未来日的收益率
                    future_day_return = (
                        future_feature_impact * 0.5 +  # 特征影响（主要因素）
                        future_market_impact * 0.3 +   # 市场影响
                        future_industry_effect * 0.2 +  # 行业效应
                        future_noise +                 # 随机噪声
                        curr_return_effect             # 当前收益率的小影响
                    )
                    
                    future_days_returns.append(future_day_return)
                
                # 计算加权未来收益率
                future_return = np.sum(weights * future_days_returns)
            else:
                # 对于接近结尾的日期，使用历史均值加噪声
                if len(returns) > 0:
                    future_return = np.mean(returns[-min(len(returns), 10):]) + np.random.normal(0, 0.02)
                else:
                    future_return = np.random.normal(0, 0.02)
            
            # 添加数据行
            row = {
                'date': dates[day],
                'stock': stock,
                'industry': industry,
                'price': price,
                'future_return': future_return
            }
            
            # 如果市场因子启用，添加到数据
            if add_market_factor and day < n_days:
                row['market_factor'] = market_factor[day]
            
            # 添加特征
            for i in range(20):
                row[f'feature_{i+1}'] = features[i]
            
            data_rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows)
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to stock data
    
    Parameters:
    df: Stock data DataFrame
    
    Returns:
    DataFrame with technical indicators
    """
    # Sort by stock and date
    df = df.sort_values(['stock', 'date'])
    
    # Ensure all stocks have the same dates
    stocks = df['stock'].unique()
    
    result_dfs = []
    
    for stock in stocks:
        stock_data = df[df['stock'] == stock].copy()
        
        # Calculate price change
        stock_data['price_change'] = stock_data['price'].diff()
        
        # Calculate return
        stock_data['return'] = stock_data['price'].pct_change()
        
        # Calculate volatility (20-day rolling std)
        stock_data['volatility'] = stock_data['return'].rolling(window=20).std().fillna(0)
        
        # Calculate moving averages
        stock_data['MA5'] = stock_data['price'].rolling(window=5).mean().fillna(stock_data['price'])
        stock_data['MA10'] = stock_data['price'].rolling(window=10).mean().fillna(stock_data['price'])
        stock_data['MA20'] = stock_data['price'].rolling(window=20).mean().fillna(stock_data['price'])
        
        # Calculate MACD
        stock_data['EMA12'] = stock_data['price'].ewm(span=12, adjust=False).mean()
        stock_data['EMA26'] = stock_data['price'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
        stock_data['MACD_signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        stock_data['MACD_hist'] = stock_data['MACD'] - stock_data['MACD_signal']
        
        # Calculate RSI
        delta = stock_data['price'].diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean().fillna(0)
        avg_loss = loss.rolling(window=14).mean().fillna(0)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        stock_data['RSI'] = stock_data['RSI'].fillna(50)  # Fill starting NaN values
        
        # Calculate Bollinger Bands
        stock_data['BB_middle'] = stock_data['price'].rolling(window=20).mean()
        rolling_std = stock_data['price'].rolling(window=20).std()
        stock_data['BB_upper'] = stock_data['BB_middle'] + (rolling_std * 2)
        stock_data['BB_lower'] = stock_data['BB_middle'] - (rolling_std * 2)
        
        # Add lag features
        for lag in range(1, 6):  # Add 5 days of lag features
            stock_data[f'price_lag_{lag}'] = stock_data['price'].shift(lag)
            stock_data[f'return_lag_{lag}'] = stock_data['return'].shift(lag)
        
        # Calculate price/moving average ratios
        stock_data['price_to_MA5'] = stock_data['price'] / stock_data['MA5']
        stock_data['price_to_MA20'] = stock_data['price'] / stock_data['MA20']
        
        # Calculate moving average crossover signals
        stock_data['MA5_cross_MA20'] = ((stock_data['MA5'] > stock_data['MA20']) & 
                                       (stock_data['MA5'].shift(1) <= stock_data['MA20'].shift(1))).astype(int) - \
                                      ((stock_data['MA5'] < stock_data['MA20']) & 
                                       (stock_data['MA5'].shift(1) >= stock_data['MA20'].shift(1))).astype(int)
        
        # Fill NaN values
        stock_data = stock_data.fillna(0)
        
        result_dfs.append(stock_data)
    
    # Merge all stock data
    result_df = pd.concat(result_dfs)
    
    return result_df

def preprocess_data(df, time_series_split=False, all_industry_dummies=None):
    """
    Enhanced data preprocessing
    
    Parameters:
    df: Stock data DataFrame
    time_series_split: Whether to use time series split instead of random split
    all_industry_dummies: Pre-computed industry dummies for consistent feature sets
    
    Returns:
    X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names, industry_dummies
    """
    # Feature selection
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
    
    # For handling industry information, we need dummies
    if all_industry_dummies is None:
        industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
    else:
        # Use pre-computed dummies to ensure consistent feature set
        temp_dummies = pd.get_dummies(df['industry'], prefix='industry')
        industry_dummies = pd.DataFrame(0, index=df.index, columns=all_industry_dummies.columns)
        for col in temp_dummies.columns:
            if col in industry_dummies.columns:
                industry_dummies[col] = temp_dummies[col]
    
    # Prepare feature matrix and target variable
    X = df[all_features].values
    y = df['future_return'].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Combine industry dummies
    X_with_industry = np.hstack([X, industry_dummies.values])
    
    # Record feature names (for model interpretation)
    feature_names = all_features + industry_dummies.columns.tolist()
    
    # Data split
    if time_series_split:
        # Sort by time
        df = df.sort_values('date')
        
        # Determine split points
        total_rows = len(df)
        train_end = int(total_rows * 0.6)
        valid_end = int(total_rows * 0.8)
        
        # Split data
        X_train = X_with_industry[:train_end]
        y_train = y[:train_end]
        
        X_valid = X_with_industry[train_end:valid_end]
        y_valid = y[train_end:valid_end]
        
        X_test = X_with_industry[valid_end:]
        y_test = y[valid_end:]
    else:
        # Random split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_with_industry, y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names, industry_dummies