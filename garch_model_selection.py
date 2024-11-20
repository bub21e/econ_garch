import pandas as pd
import numpy as np
from arch import arch_model
from itertools import product
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_features(df):
    """
    计算用于GARCH模型的特征
    """
    # 计算对数收益率
    df['log_return'] = np.log(df['收盘价_Clpr']/df['收盘价_Clpr'].shift(1))
    
    # 计算日内波动率
    df['daily_volatility'] = (np.log(df['最高价_Hipr']) - np.log(df['最低价_Lopr'])) / np.sqrt(2)
    
    # 计算成交量变化
    df['volume_change'] = np.log(df['成交量_Trdvol']/df['成交量_Trdvol'].shift(1))
    
    # 计算开盘收盘价差
    df['open_close_diff'] = np.log(df['收盘价_Clpr']/df['开盘价_Oppr'])
    
    # 计算市值变化
    df['market_value_change'] = np.log(df['日总市值(元)_Dmc']/df['日总市值(元)_Dmc'].shift(1))
    df['float_value_change'] = np.log(df['日流通市值(元)_Dtmv']/df['日流通市值(元)_Dtmv'].shift(1))
    
    # 删除缺失值
    df = df.dropna()
    
    return df

def calculate_mse(model_fit, returns):
    """
    计算模型的MSE
    """
    forecasts = model_fit.forecast(horizon=1) # 遍历预测
    forecast_values = forecasts.mean.values.flatten() # 转一维
    mse = np.mean((returns - forecast_values) ** 2)

    return mse

def generate_evaluation(df, max_p=3, max_q=3):
    """
    选择最优的GARCH模型阶数，考虑外生变量
    """
    exog_vars = [
        'daily_volatility', 'volume_change', 'open_close_diff',
        'market_value_change', 'float_value_change'
    ]
    X = df[exog_vars].values
    returns = df['log_return'].values
    
    print(f"Returns shape: {returns.shape}") # 收益率数组维度
    print(f"Exogenous variables shape: {X.shape}") # 包含外生变量数组维度
    
    # 标准化外生变量
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    results = []
    
    # 拟合模型
    for p, q in product(range(1, max_p + 1), range(1, max_q + 1)):
        try:
            # 基础模型
            model = arch_model(returns, vol='Garch', p=p, q=q, dist='studentst')
            model_fit = model.fit(disp='off', show_warning=False)
            
            mse = calculate_mse(model_fit, returns)
            
            results.append({
                'p': p,
                'q': q,
                'model_type': 'base',
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'log_likelihood': model_fit.loglikelihood,
                'mse': mse
            })
            
            # 带外生变量的模型
            model_with_x = arch_model(returns, vol='Garch', p=p, q=q, dist='studentst', x=X)
            model_fit_with_x = model_with_x.fit(disp='off', show_warning=False)
            
            mse_x = calculate_mse(model_fit_with_x, returns)
            
            results.append({
                'p': p,
                'q': q,
                'model_type': 'with_exog',
                'aic': model_fit_with_x.aic,
                'bic': model_fit_with_x.bic,
                'log_likelihood': model_fit_with_x.loglikelihood,
                'mse': mse_x
            })
            
        except Exception as e:
            print(f"GARCH({p},{q})模型拟合失败: {str(e)}")
            continue
    
    if not results:
        raise ValueError("所有GARCH模型拟合都失败了，请检查数据")
    
    results_df = pd.DataFrame(results)
    return results_df 



def select_best_model(results_df):
    """
    选择最佳模型：
    1. 筛选MSE不大于最小MSE 5%的模型
    2. 在这些模型中选择AIC最小的模型
    """
    # 计算MSE阈值
    mse_threshold = results_df['mse'].min() * 1.05
    
    # 筛选MSE合格的模型
    qualified_models = results_df[results_df['mse'] <= mse_threshold].copy()

    # 在这些模型中选择AIC最小的模型
    return qualified_models.loc[qualified_models['aic'].idxmin()]



# test function
# def main():
#     try:
#         # 读取数据
#         print("读取数据...")
#         df = pd.read_csv('processed_stock_data.csv')
#         df['日期_Date'] = pd.to_datetime(df['日期_Date'])
        
#         # 选择市值最大的几只股票
#         latest_date = df['日期_Date'].max()
#         top_stocks = df[df['日期_Date'] == latest_date].nlargest(5, '日总市值(元)_Dmc')['股票代码_Stkcd'].tolist()
        
#         print("\n选择的股票代码:", top_stocks)
        
#         for stock_code in top_stocks:
#             try:
#                 print(f"\n{'='*50}")
#                 print(f"分析股票代码: {stock_code}")
                
#                 # 获取单个股票的数据
#                 stock_data = df[df['股票代码_Stkcd'] == stock_code].copy()
#                 stock_data = stock_data.sort_values('日期_Date')
                
#                 # 计算特征
#                 stock_data = calculate_features(stock_data)
                
#                 # 选择最优阶数并获取结果
#                 results_df = generate_evaluation(stock_data)
                
#                 # 选择最优模型
#                 best_model = select_best_model(results_df)
                
#                 # 输出最优模型信息
#                 print("\n最优模型信息:")
#                 print(f"GARCH阶数: ({best_model['p']},{best_model['q']})")
#                 print(f"模型类型: {best_model['model_type']}")
#                 print(f"AIC: {best_model['aic']:.4f}")
#                 print(f"BIC: {best_model['bic']:.4f}")
#                 print(f"对数似然值: {best_model['log_likelihood']:.4f}")
#                 print(f"MSE: {best_model['mse']:.6f}")
                
#             except Exception as e:
#                 print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
#                 continue
            
#     except Exception as e:
#         print(f"发生错误: {str(e)}")
#         import traceback
#         print(traceback.format_exc())

# if __name__ == "__main__":
#     main()