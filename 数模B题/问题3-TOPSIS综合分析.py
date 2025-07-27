import pandas as pd
import numpy as np

# 设置pandas显示选项，优化表格输出
pd.set_option('display.unicode.east_asian_width', True)  # 自动调整列宽
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)  # 调整显示宽度

# 定义熵权法计算权重的函数
def entropy_weight(data):
    """
    使用熵权法计算指标权重
    
    参数:
    data (DataFrame): 包含决策矩阵的数据框
    
    返回:
    tuple: (各指标的权重, 各指标的熵值)
    """
    # 归一化处理
    norm_df = data / np.sqrt(np.sum(data**2, axis=0))
    
    # 计算概率矩阵
    p = norm_df / norm_df.sum(axis=0)
    
    # 避免log(0)的情况
    p = p.replace(0, 1e-10)
    
    # 计算熵值
    e = -1 / np.log(len(p)) * np.sum(p * np.log(p), axis=0)
    
    # 计算差异系数
    d = 1 - e
    
    # 计算权重
    weights = d / np.sum(d)
    
    return weights.tolist(), e.tolist()

# 定义TOPSIS分析函数
def topsis_analysis(data, weights, criteria_types):
    """
    执行TOPSIS分析
    
    参数:
    data (DataFrame): 包含决策矩阵的数据框
    weights (list): 各指标的权重
    criteria_types (list): 各指标的类型，'效益型'或'成本型'
    
    返回:
    DataFrame: 包含TOPSIS分析结果的数据框
    """
    # 复制数据
    df = data.copy()
    
    # 归一化处理
    norm_df = df / np.sqrt(np.sum(df**2, axis=0))
    
    # 加权归一化
    weighted_norm_df = norm_df * weights
    
    # 确定理想解和负理想解
    ideal_best = weighted_norm_df.apply(
        lambda x: max(x) if criteria_types[list(weighted_norm_df.columns).index(x.name)] == '效益型' 
        else min(x), axis=0
    )
    ideal_worst = weighted_norm_df.apply(
        lambda x: min(x) if criteria_types[list(weighted_norm_df.columns).index(x.name)] == '效益型' 
        else max(x), axis=0
    )
    
    # 计算与理想解和负理想解的距离
    s_best = np.sqrt(np.sum((weighted_norm_df - ideal_best)**2, axis=1))
    s_worst = np.sqrt(np.sum((weighted_norm_df - ideal_worst)**2, axis=1))
    
    # 计算相对接近度
    performance_score = s_worst / (s_best + s_worst)
    
    # 添加结果到数据框
    result_df = data.copy()
    result_df['理想解距离'] = s_best
    result_df['负理想解距离'] = s_worst
    result_df['相对接近度'] = performance_score
    result_df['排名'] = result_df['相对接近度'].rank(ascending=False)
    
    return result_df

# 创建疗效数据（更新为改善指数）
efficacy_data = {
    '组别': [1, 2, 3],
    '第一月不良症状发生率': [1.69, 1.60, 1.56],
    '第三月不良症状发生率': [0.92, 0.86, 0.76],
    '第六月不良症状发生率': [0.51, 0.48, 0.43],
    '第十二月不良症状发生率': [0.28, 0.27, 0.20],
    '总体恢复率': [0.91, 0.88, 0.91],
    '复发率': [0.25, 0.35, 0.20],
    '改善指数': [0.6756, 0.6712, 0.7341]  
}

df = pd.DataFrame(efficacy_data)
df.set_index('组别', inplace=True)

# 使用熵权法计算权重和熵值
weights, entropy = entropy_weight(df)
 
# 输出熵权法计算结果
print("熵权法计算结果:")
for i, col in enumerate(df.columns):
    print(f"{col}: 熵值 = {entropy[i]:.4f}, 权重 = {weights[i]:.4f}")

# 定义指标类型（对应更新后的指标）
criteria_types = [
    '成本型',  # 第一月不良症状发生率（越低越好）
    '成本型',  # 第三月不良症状发生率（越低越好）
    '成本型',  # 第六月不良症状发生率（越低越好）
    '成本型',  # 第十二月不良症状发生率（越低越好）
    '效益型',  # 总体恢复率（越高越好）
    '成本型',  # 复发率（越低越好）
    '效益型'   # 改善指数（越高越好）
]

# 执行TOPSIS分析
result = topsis_analysis(df, weights, criteria_types)

# 分块输出结果表格（避免过长）
print("\n三种抗抑郁药物的疗效TOPSIS分析结果（基础指标）：")
print(result.iloc[:, :5])  # 输出前5列基础指标
print("\n三种抗抑郁药物的疗效TOPSIS分析结果（剩余指标及分析结果）：")
print(result.iloc[:, 5:])  # 输出剩余指标及TOPSIS分析结果