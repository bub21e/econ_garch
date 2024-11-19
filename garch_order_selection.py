import pandas as pd
import numpy as np
from arch import arch_model

from itertools import product # 笛卡尔积
import matplotlib.pyplot as plt # 绘图
import seaborn as sns # 绘图
from sklearn.preprocessing import StandardScaler # 标准化
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import os

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
    
    # 检查数据维度
    print(f"Returns shape: {returns.shape}")
    print(f"Exogenous variables shape: {X.shape}")
    
    # 标准化外生变量
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    results = []
    
    # 先拟合基础模型
    print("\n拟合基础GARCH模型...")
    for p, q in product(range(1, max_p + 1), range(1, max_q + 1)):
        try:
            print(f"\n尝试拟合基础 GARCH({p},{q}) 模型...")
            model = arch_model(returns, 
                             vol='Garch', 
                             p=p, 
                             q=q, 
                             dist='studentst')
            model_fit = model.fit(disp='off', show_warning=False)
            
            # 计算MSE
            forecasts = model_fit.forecast(horizon=1)
            forecast_mean = forecasts.mean.values[-1][0]  # 获取最后一个预测值
            actual = returns[-1]  # 获取最后一个实际值
            mse = (actual - forecast_mean) ** 2
            
            # 获取基础模型评价指标
            results.append({
                'p': p,
                'q': q,
                'model_type': 'base',
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'log_likelihood': model_fit.loglikelihood,
                'mse': mse
            })
            print(f"基础 GARCH({p},{q}) 模型拟合成功")
            
            # 尝试添加外生变量
            print(f"\n尝试拟合带外生变量的 GARCH({p},{q}) 模型...")
            model_with_x = arch_model(returns, 
                                    vol='Garch', 
                                    p=p, 
                                    q=q, 
                                    dist='studentst',
                                    x=X)
            model_fit_with_x = model_with_x.fit(disp='off', show_warning=False)
            
            # 计算带外生变量模型的MSE
            forecasts_x = model_fit_with_x.forecast(horizon=1)
            forecast_mean_x = forecasts_x.mean.values[-1][0]
            mse_x = (actual - forecast_mean_x) ** 2
            
            # 获取带外生变量模型的评价指标
            results.append({
                'p': p,
                'q': q,
                'model_type': 'with_exog',
                'aic': model_fit_with_x.aic,
                'bic': model_fit_with_x.bic,
                'log_likelihood': model_fit_with_x.loglikelihood,
                'mse': mse_x
            })
            print(f"带外生变量的 GARCH({p},{q}) 模型拟合成功")
            
        except Exception as e:
            print(f"GARCH({p},{q})模型拟合失败: {str(e)}")
            continue
    
    if not results:
        raise ValueError("所有GARCH模型拟合都失败了，请检查数据")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.reset_index(drop=True)  # 重置索引避免重复
    
    # 打印模型比较结果
    print("\n模型比较结果：")
    print("\n基础模型：")
    print(results_df[results_df['model_type'] == 'base'].sort_values('aic'))
    print("\n带外生变量的模型：")
    print(results_df[results_df['model_type'] == 'with_exog'].sort_values('aic'))
    
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
    
    # 确保索引唯一
    results_df = results_df.reset_index(drop=True)
    top_3_models = results_df.nsmallest(3, 'aic')
    
    print("\n前三个最优模型的详细比较：")
    print("-" * 50)
    
    for idx, model_info in top_3_models.iterrows():
        p, q = int(model_info['p']), int(model_info['q'])
        
        # 根据模型类型决定是否使用外生变量
        if model_info['model_type'] == 'with_exog':
            model = arch_model(returns, vol='Garch', p=p, q=q, dist='studentst', x=X)
        else:
            model = arch_model(returns, vol='Garch', p=p, q=q, dist='studentst')
            
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
        
        # 如果是带外生变量的模型才分析外生变量影响
        if model_info['model_type'] == 'with_exog':
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

def plot_model_diagnostics(model_fit, stock_code, p, q, model_type):
    """
    绘制模型诊断图
    """
    plt.figure(figsize=(15, 10))
    
    # 1. 标准化残差
    resid = model_fit.resid/model_fit.conditional_volatility
    
    # 2. 残差分布图
    plt.subplot(2, 2, 1)
    sns.histplot(resid, kde=True)
    plt.title('标准化残差分布')
    
    # 3. QQ图
    plt.subplot(2, 2, 2)
    from scipy import stats
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title('QQ图')
    
    # 4. 波动率图
    plt.subplot(2, 2, 3)
    plt.plot(model_fit.conditional_volatility)
    plt.title('条件波动率')
    
    # 5. 自相关图
    plt.subplot(2, 2, 4)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid**2, lags=40)
    plt.title('残差平方自相关图')
    
    plt.tight_layout()
    plt.savefig(f'model_diagnostics_{stock_code}_p{p}_q{q}_{model_type}.png')
    plt.close()

def calculate_model_metrics(model_fit, returns):
    """
    计算模型的各种评价指标
    """
    resid = model_fit.resid/model_fit.conditional_volatility
    
    metrics = {
        'AIC': model_fit.aic,
        'BIC': model_fit.bic,
        'Log-Likelihood': model_fit.loglikelihood,
        'Jarque-Bera': stats.jarque_bera(resid)[1],  # 正态性检验
        'Ljung-Box': stats.acorr_ljungbox(resid**2, lags=[10])[1][0],  # 自相关检验
        'Durbin-Watson': durbin_watson(resid)  # 序列相关检验
    }
    
    return metrics

def create_output_directory(stock_code):
    """
    为每只股票创建输出目录
    """
    output_dir = f'output_{stock_code}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_volatility_clustering(returns, stock_code):
    """
    绘制波动率聚集效应图
    """
    plt.figure(figsize=(12, 6))
    plt.plot(returns**2)
    plt.title(f'股票{stock_code}的波动率聚集效应')
    plt.xlabel('时间')
    plt.ylabel('平方收益率')
    plt.tight_layout()
    plt.savefig(f'output_{stock_code}/volatility_clustering.png')
    plt.close()

def plot_returns_distribution(returns, stock_code):
    """
    绘制收益率分布图
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图和核密度估计
    sns.histplot(returns, kde=True, stat='density')
    
    # 添加正态分布曲线作为参考
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')
    
    plt.title(f'股票{stock_code}的收益率分布')
    plt.xlabel('收益率')
    plt.ylabel('密度')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output_{stock_code}/returns_distribution.png')
    plt.close()

def generate_summary_report(summary_results, output_file='model_summary_report.txt'):
    """
    生成模型汇总报告
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("GARCH模型分析汇总报告\n")
        f.write("=" * 50 + "\n\n")
        
        for result in summary_results:
            f.write(f"股票代码: {result['stock_code']}\n")
            f.write(f"最优模型: {result['best_model']}\n")
            f.write(f"模型类型: {result['model_type']}\n")
            f.write(f"AIC: {result['AIC']:.2f}\n")
            f.write(f"BIC: {result['BIC']:.2f}\n")
            f.write(f"对数似然值: {result['Log-Likelihood']:.2f}\n")
            f.write(f"Jarque-Bera检验p值: {result['Jarque-Bera']:.4f}\n")
            f.write(f"Ljung-Box检验p值: {result['Ljung-Box']:.4f}\n")
            f.write(f"Durbin-Watson统计量: {result['Durbin-Watson']:.4f}\n")
            f.write("\n" + "-" * 50 + "\n\n")

def plot_rolling_statistics(returns, stock_code, window=20):
    """
    绘制滚动统计图
    """
    plt.figure(figsize=(15, 10))
    
    # 计算滚动统计量
    rolling_mean = pd.Series(returns).rolling(window=window).mean()
    rolling_std = pd.Series(returns).rolling(window=window).std()
    rolling_skew = pd.Series(returns).rolling(window=window).skew()
    rolling_kurt = pd.Series(returns).rolling(window=window).kurt()
    
    # 绘制滚动均值
    plt.subplot(2, 2, 1)
    plt.plot(rolling_mean)
    plt.title(f'滚动均值 (窗口={window})')
    plt.xlabel('时间')
    plt.ylabel('均值')
    
    # 绘制滚动标准差
    plt.subplot(2, 2, 2)
    plt.plot(rolling_std)
    plt.title(f'滚动标准差 (窗口={window})')
    plt.xlabel('时间')
    plt.ylabel('标准差')
    
    # 绘制滚动偏度
    plt.subplot(2, 2, 3)
    plt.plot(rolling_skew)
    plt.title(f'滚动偏度 (窗口={window})')
    plt.xlabel('时间')
    plt.ylabel('偏度')
    
    # 绘制滚动峰度
    plt.subplot(2, 2, 4)
    plt.plot(rolling_kurt)
    plt.title(f'滚动峰度 (窗口={window})')
    plt.xlabel('时间')
    plt.ylabel('峰度')
    
    plt.tight_layout()
    plt.savefig(f'output_{stock_code}/rolling_statistics.png')
    plt.close()

def perform_arch_test(returns, lags=10):
    """
    执行ARCH效应检验
    """
    from statsmodels.stats.diagnostic import het_arch
    
    # ARCH LM检验
    lm_stat, lm_pval, f_stat, f_pval = het_arch(returns, nlags=lags)
    
    return {
        'LM统计量': lm_stat,
        'LM p值': lm_pval,
        'F统计量': f_stat,
        'F p值': f_pval
    }

def calculate_descriptive_stats(returns):
    """
    计算描述性统计量
    """
    stats_dict = {
        '样本量': len(returns),
        '均值': np.mean(returns),
        '标准差': np.std(returns),
        '偏度': stats.skew(returns),
        '峰度': stats.kurtosis(returns),
        'Jarque-Bera统计量': stats.jarque_bera(returns)[0],
        'Jarque-Bera p值': stats.jarque_bera(returns)[1]
    }
    return stats_dict

def generate_detailed_report(stock_code, stock_data, model_fit, metrics, arch_test_results, desc_stats):
    """
    生成详细的分析报告
    """
    output_dir = f'output_{stock_code}'
    report_file = f'{output_dir}/detailed_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"股票{stock_code}的GARCH模型详细分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. 数据基本信息
        f.write("1. 数据基本信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"数据时间范围: {stock_data['日期_Date'].min()} 到 {stock_data['日期_Date'].max()}\n")
        f.write(f"数据条数: {len(stock_data)}\n\n")
        
        # 2. 描述性统计
        f.write("2. 收益率描述性统计\n")
        f.write("-" * 30 + "\n")
        for key, value in desc_stats.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")
        
        # 3. ARCH效应检验
        f.write("3. ARCH效应检验结果\n")
        f.write("-" * 30 + "\n")
        for key, value in arch_test_results.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")
        
        # 4. 模型参数
        f.write("4. 模型参数估计\n")
        f.write("-" * 30 + "\n")
        f.write("参数估计值：\n")
        f.write(str(model_fit.params) + "\n\n")
        f.write("参数显著性（p值）：\n")
        f.write(str(model_fit.pvalues) + "\n\n")
        
        # 5. 模型评价指标
        f.write("5. 模型评价指标\n")
        f.write("-" * 30 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")
        
        # 6. 预测评估
        f.write("6. 预测评估\n")
        f.write("-" * 30 + "\n")
        forecasts = model_fit.forecast(horizon=5)
        f.write("未来5天的波动率预测：\n")
        f.write(str(forecasts.variance.values[-1]) + "\n")

def main():
    try:
        # 读取处理后的数据
        print("读取数据...")
        df = pd.read_csv('processed_stock_data.csv')
        df['日期_Date'] = pd.to_datetime(df['日期_Date'])
        
        # 创建结果汇总DataFrame
        summary_results = []
        
        # 选择市值最大的几只股票进行分析
        latest_date = df['日期_Date'].max()
        top_stocks = df[df['日期_Date'] == latest_date].nlargest(5, '日总市值(元)_Dmc')['股票代码_Stkcd'].tolist()
        
        for stock_code in top_stocks:
            try:
                print(f"\n分析股票代码: {stock_code}")
                
                # 创建输出目录
                output_dir = create_output_directory(stock_code)
                
                # 获取单个股票的数据
                stock_data = df[df['股票代码_Stkcd'] == stock_code].copy()
                stock_data = stock_data.sort_values('日期_Date')
                
                # 计算特征
                stock_data = calculate_features(stock_data)
                
                # 绘制初步分析图
                plot_volatility_clustering(stock_data['log_return'].values, stock_code)
                plot_returns_distribution(stock_data['log_return'].values, stock_code)
                
                # 选择最优阶数和模型拟合
                results_df = select_order(stock_data)
                
                if len(results_df) > 0:
                    # 获取最优模型
                    best_model = results_df.loc[results_df['aic'].idxmin()]
                    p, q = int(best_model['p']), int(best_model['q'])
                    
                    # 拟合最优模型
                    returns = stock_data['log_return'].values
                    if best_model['model_type'] == 'with_exog':
                        exog_vars = ['daily_volatility', 'volume_change', 'open_close_diff',
                                   'market_value_change', 'float_value_change']
                        X = StandardScaler().fit_transform(stock_data[exog_vars])
                        model = arch_model(returns, vol='Garch', p=p, q=q, dist='studentst', x=X)
                    else:
                        model = arch_model(returns, vol='Garch', p=p, q=q, dist='studentst')
                    
                    model_fit = model.fit(disp='off')
                    
                    # 绘制诊断图
                    plot_model_diagnostics(model_fit, stock_code, p, q, best_model['model_type'])
                    
                    # 计算评价指标
                    metrics = calculate_model_metrics(model_fit, returns)
                    
                    # 添加新的分析
                    plot_rolling_statistics(returns, stock_code)
                    arch_test_results = perform_arch_test(returns)
                    desc_stats = calculate_descriptive_stats(returns)
                    
                    # 生成详细报告
                    generate_detailed_report(
                        stock_code, 
                        stock_data, 
                        model_fit, 
                        metrics, 
                        arch_test_results, 
                        desc_stats
                    )
                    
                    # 添加到汇总结果
                    summary_results.append({
                        'stock_code': stock_code,
                        'best_model': f"GARCH({p},{q})",
                        'model_type': best_model['model_type'],
                        **metrics
                    })
                    
                    # 保存结果到指定目录
                    results_df.to_csv(f'{output_dir}/garch_order_selection_results.csv')
                
            except Exception as e:
                print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
                continue
        
        # 生成汇总报告
        generate_summary_report(summary_results)
        pd.DataFrame(summary_results).to_csv('garch_model_summary.csv', index=False)
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 