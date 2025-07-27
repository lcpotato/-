import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 以下是模型合理性检验代码
def sensitivity_analysis(data, criteria_types, iterations=50, perturbation=0.1):
    """
    敏感性分析：测试权重微小变化对结果排序的影响
    
    参数:
    data: 原始数据
    criteria_types: 指标类型列表
    iterations: 随机扰动次数
    perturbation: 扰动幅度
    
    返回:
    稳定性得分(0-1之间，越高表示结果越稳定)
    """
    # 获取原始排名
    original_ranking = result['排名'].copy()
    
    # 记录每次扰动后排名变化的数量
    ranking_changes = []
    
    for _ in range(iterations):
        # 生成随机扰动
        random_weights = np.array(weights) * (1 + np.random.uniform(-perturbation, perturbation, len(weights)))
        # 重新归一化权重
        random_weights = random_weights / np.sum(random_weights)
        
        # 使用扰动后的权重进行TOPSIS分析
        perturbed_result = topsis_analysis(data, random_weights, criteria_types)
        perturbed_ranking = perturbed_result['排名']
        
        # 计算排名变化数量
        ranking_changes.append(np.sum(original_ranking != perturbed_ranking))
    
    # 计算稳定性得分：排名不变的比例
    stability_score = 1 - np.mean(ranking_changes) / len(data)
    
    return stability_score

def correlation_analysis(data):
    """
    指标相关性分析
    
    参数:
    data: 原始数据
    
    返回:
    相关系数矩阵和高相关性指标对
    """
    # 计算相关系数矩阵
    corr_matrix = data.corr()
    
    # 找出高相关性指标对(相关系数绝对值大于0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    return corr_matrix, high_corr_pairs

def rank_consistency_test(data, criteria_types, n_samples=30, sample_size=0.8):
    """
    结果稳定性测试：抽样检验排名一致性
    
    参数:
    data: 原始数据
    criteria_types: 指标类型列表
    n_samples: 抽样次数
    sample_size: 每次抽样比例
    
    返回:
    一致性得分(0-1之间，越高表示结果越稳定)
    """
    # 获取原始排名
    original_ranking = result['排名'].copy()
    
    # 记录每次抽样后的Kendall秩相关系数
    kendall_taus = []
    
    for _ in range(n_samples):
        # 随机抽样（使用数据框的实际索引而非位置索引）
        sample_indices = np.random.choice(data.index, size=int(len(data)*sample_size), replace=False)
        sample_data = data.loc[sample_indices]
        
        # 使用原始权重对抽样数据进行TOPSIS分析
        sample_result = topsis_analysis(sample_data, weights, criteria_types)
        sample_ranking = sample_result['排名']
        
        # 确保比较相同索引的排名
        original_ranking_subset = original_ranking.loc[sample_indices]
        
        # 计算Kendall秩相关系数
        tau, _ = stats.kendalltau(original_ranking_subset, sample_ranking)
        kendall_taus.append(tau)
    
    # 返回平均Kendall秩相关系数作为一致性得分
    return np.mean(kendall_taus)


def plot_weights_comparison():
    """绘制各指标权重对比图"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df.columns, weights, color='skyblue')
    plt.xlabel('指标')
    plt.ylabel('权重')
    plt.title('各指标权重分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.show()

def plot_correlation_heatmap(corr_matrix):
    """绘制相关性热图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('指标相关性热图')
    
    # 添加数值标签
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                     ha='center', va='center',fontsize=11, color='black' if abs(corr_matrix.iloc[i, j]) < 0.7 else 'white')
    
    plt.tight_layout()
    plt.show()

# 执行合理性检验
stability_score = sensitivity_analysis(df, criteria_types)
consistency_score = rank_consistency_test(df, criteria_types)
corr_matrix, high_corr_pairs = correlation_analysis(df)

# 输出检验结果
print("\n===== TOPSIS模型合理性检验结果 =====")
print(f"1. 敏感性分析稳定性得分: {stability_score:.4f}")
print(f"2. 抽样稳定性检验一致性得分: {consistency_score:.4f}")

print("\n3. 指标相关性分析:")
if high_corr_pairs:
    print("   高相关性指标对:")
    for pair in high_corr_pairs:
        print(f"   {pair[0]} 和 {pair[1]}: 相关系数 = {pair[2]:.4f}")
else:
    print("   未发现高相关性指标对")

# 计算并输出变异系数(CV)，衡量指标离散程度
cv = df.std() / df.mean()
print("\n4. 各指标变异系数:")
for col in cv.index:
    print(f"   {col}: {cv[col]:.4f}")

# 绘制权重分布图和相关性热图
plot_weights_comparison()
plot_correlation_heatmap(corr_matrix)