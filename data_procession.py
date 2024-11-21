import pandas as pd
import os
import glob
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_price_data(price_path):
    """
    加载价格数据文件
    """
    print("\n开始加载价格数据文件...")
    
    price_files = glob.glob(os.path.join(price_path, "*.xls"))
    
    if not price_files:
        raise FileNotFoundError(f"在 {price_path} 目录下没有找到价格数据文件")
    
    print(f"找到 {len(price_files)} 个价格数据文件")
    
    price_list = []
    for file in tqdm(price_files, desc="读取价格文件"):
        try:
            df = pd.read_excel(file, engine='xlrd')
            
            # 验证价格数据必要的列
            required_columns = [
                '股票代码_Stkcd', '日期_Date', 
                '开盘价_Oppr', '最高价_Hipr',
                '最低价_Lopr', '收盘价_Clpr', 
                '成交量_Trdvol'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告: 价格文件 {os.path.basename(file)} 缺少列: {missing_columns}")
                continue
            
            print(f"\n成功读取价格文件: {os.path.basename(file)}")
            print(f"数据形状: {df.shape}")
            price_list.append(df)
            
        except Exception as e:
            print(f"\n错误: 读取价格文件 {os.path.basename(file)} 时出错:")
            print(f"错误信息: {str(e)}")
            continue
    
    if not price_list:
        raise ValueError("没有成功读取任何价格数据文件")
    
    print("\n合并价格数据文件...")
    price_combined = pd.concat(price_list, ignore_index=True)
    
    return price_combined

def process_price_data(df):
    """
    处理价格数据
    """
    print("\n处理价格数据...")
    
    df['日期_Date'] = pd.to_datetime(df['日期_Date'], format='%Y%m%d')
    
    # 按股票代码和日期排序
    df = df.sort_values(['股票代码_Stkcd', '日期_Date'])
    
    # 删除重复数据
    duplicates = df.duplicated(['股票代码_Stkcd', '日期_Date'])
    if duplicates.any():
        print(f"发现 {duplicates.sum()} 条重复记录，正在删除...")
        df = df.drop_duplicates(['股票代码_Stkcd', '日期_Date'])
    
    # 数据有效性检查
    df['数据有效'] = (
        (df['收盘价_Clpr'] > 0) & 
        (df['开盘价_Oppr'] > 0) & 
        (df['最高价_Hipr'] > 0) & 
        (df['最低价_Lopr'] > 0) & 
        (df['成交量_Trdvol'] >= 0)
    )
    
    invalid_count = (~df['数据有效']).sum()
    if invalid_count > 0:
        print(f"发现 {invalid_count} 条无效数据记录，正在删除...")
        df = df[df['数据有效']].drop('数据有效', axis=1)
    
    return df

def analyze_data_quality(df):
    """
    分析数据质量
    """
    print("\n数据质量分析:")
    
    print("\n1. 基本信息:")
    print(f"总记录数: {len(df):,}")
    print(f"股票数量: {df['股票代码_Stkcd'].nunique():,}")
    print(f"时间范围: {df['日期_Date'].min()} 到 {df['日期_Date'].max()}")
    
    print("\n2. 数据完整性:")
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("\n存在缺失值的列:")
        print(missing_data[missing_data > 0])
    else:
        print("所有列数据完整，无缺失值")
    
    print("\n3. 交易日统计:")
    trading_days = df.groupby('股票代码_Stkcd')['日期_Date'].nunique()
    print(f"平均交易天数: {trading_days.mean():.0f}")
    print(f"最少交易天数: {trading_days.min()}")
    print(f"最多交易天数: {trading_days.max()}")
    
    return trading_days

def main():
    try:
        # 设置数据路径
        price_path = r"C:\Users\Administrator\Desktop\data_1\dtk"
        
        # 加载价格数据
        price_df = load_price_data(price_path)
        
        # 处理价格数据
        processed_df = process_price_data(price_df)
        
        # 分析数据质量
        trading_days = analyze_data_quality(processed_df)
        
        # 保存处理后的数据
        print("\n保存处理后的数据...")
        processed_df.to_csv('processed_stock_data.csv', index=False)
        
        print("\n数据处理完成！")
        
    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 