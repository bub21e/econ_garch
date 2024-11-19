import pandas as pd
import numpy as np
from arch import arch_model
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
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

def select_order(df, max_p=3, max_q=3):
    """
    选择最优的GARCH模型阶数，考虑外生变量
    """
    # 准备外生变量
    exog_vars = [
        'daily_volatility', 'volume_change', 'open_close_diff',
        'market_value_change', 'float_value_change'
    ]
    X = df[exog_vars].values
    returns = df['log_return'].values
    
    # 标准化外生变量
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    results = []
    
    for p, q in product(range(1, max_p + 1), range(1, max_q + 1)):
        try:
            model = arch_model(returns, 
                             vol='Garch', 
                             p=p, 
                             q=q, 
                             dist='studentst',
                             x=X)
            model_fit = model.fit(disp='off')
            
            # 获取模型评价指标
            aic = model_fit.aic
            bic = model_fit.bic
            log_likelihood = model_fit.loglikelihood
            
            # 计算预测误差
            forecasts = model_fit.forecast(horizon=5)
            mse = np.mean((returns[5:] - forecasts.mean[:-5].values) ** 2)
            
            results.append({
                'p': p,
                'q': q,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'mse': mse
            })
            
        except Exception as e:
            print(f"GARCH({p},{q})模型拟合失败: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df

def compare_models(df, results_df):
    """
    比较不同阶数模型的表现
    """
    exog_vars = [
        'daily_volatility', 'volume_change', 'open_close_diff',
        'market_value_change', 'float_value_change'
    ]
    X = StandardScaler().fit_transform(df[exog_vars])
    returns = df['log_return'].values
    
    top_3_models = results_df.nsmallest(3, 'aic')
    
    print("\n前三个最优模型的详细比较：")
    print("-" * 50)
    
    for _, model_info in top_3_models.iterrows():
        p, q = int(model_info['p']), int(model_info['q'])
        model = arch_model(returns, 
                         vol='Garch', 
                         p=p, 
                         q=q, 
                         dist='studentst',
                         x=X)
        results = model.fit(disp='off')
        
        print(f"\nGARCH({p},{q})模型:")
        print("\n参数估计：")
        print(results.params)
        print("\n参数显著性：")
        print(results.pvalues)
        print("\n模型评价指标：")
        print(f"AIC: {results.aic:.2f}")
        print(f"BIC: {results.bic:.2f}")
        print(f"对数似然值: {results.loglikelihood:.2f}")
        print(f"MSE: {model_info['mse']:.6f}")
        
        # 分析外生变量的影响
        print("\n外生变量影响：")
        for var, coef, pval in zip(exog_vars, 
                                 results.params[1:len(exog_vars)+1],
                                 results.pvalues[1:len(exog_vars)+1]):
            significance = "显著" if pval < 0.05 else "不显著"
            print(f"{var}: 系数 = {coef:.4f}, p值 = {pval:.4f} ({significance})")
        
        print("-" * 50)

def plot_order_comparison(results_df):
    """
    绘制不同阶数模型的AIC和BIC比较图
    """
    plt.figure(figsize=(15, 6))
    
    # AIC热力图
    plt.subplot(1, 2, 1)
    aic_pivot = results_df.pivot(index='p', columns='q', values='aic')
    sns.heatmap(aic_pivot, annot=True, fmt='.0f', cmap='YlOrRd_r')
    plt.title('AIC值热力图')
    plt.xlabel('q (GARCH阶数)')
    plt.ylabel('p (ARCH阶数)')
    
    # BIC热力图
    plt.subplot(1, 2, 2)
    bic_pivot = results_df.pivot(index='p', columns='q', values='bic')
    sns.heatmap(bic_pivot, annot=True, fmt='.0f', cmap='YlOrRd_r')
    plt.title('BIC值热力图')
    plt.xlabel('q (GARCH阶数)')
    plt.ylabel('p (ARCH阶数)')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 读取处理后的数据
        print("读取数据...")
        df = pd.read_csv('processed_stock_data.csv')
        df['日期_Date'] = pd.to_datetime(df['日期_Date'])
        
        # 选择市值最大的几只股票进行分析
        latest_date = df['日期_Date'].max()
        top_stocks = df[df['日期_Date'] == latest_date].nlargest(5, '日总市值(元)_Dmc')['股票代码_Stkcd'].tolist()
        
        for stock_code in top_stocks:
            print(f"\n分析股票代码: {stock_code}")
            
            # 获取单个股票的数据
            stock_data = df[df['股票代码_Stkcd'] == stock_code].copy()
            stock_data = stock_data.sort_values('日期_Date')
            
            # 计算特征
            stock_data = calculate_features(stock_data)
            
            # 选择最优阶数
            results_df = select_order(stock_data)
            
            # 输出结果
            best_model = results_df.loc[results_df['aic'].idxmin()]
            print(f"\n基于AIC的最优模型: GARCH({int(best_model['p'])},{int(best_model['q'])})")
            print(f"AIC: {best_model['aic']:.2f}")
            print(f"BIC: {best_model['bic']:.2f}")
            print(f"MSE: {best_model['mse']:.6f}")
            
            # 比较模型
            compare_models(stock_data, results_df)
            
            # 绘制结果比较图
            plot_order_comparison(results_df)
            
            # 保存结果
            results_df.to_csv(f'garch_order_selection_results_{stock_code}.csv')
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 