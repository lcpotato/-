import pandas as pd
import numpy as np
from scipy import stats

# 读取文件
excel_file1 = pd.ExcelFile(r"C:\Users\22805\Desktop\附件1：两个院临床受试者及抑郁症的基本数据.xlsx")
excel_file2 = pd.ExcelFile(r"C:\Users\22805\Desktop\附件2：两个医院随访的抗抑郁药使用后主诉情况.xlsx")

# 读取工作表并处理复合表头（新增去重逻辑）
def load_and_clean_data(file, sheet_name, is_basic=True, hospital_name=None):
    df = file.parse(sheet_name, header=None)
    
    if is_basic:
        df.columns = [
            '序号', '组别', '年龄（岁）', 
            '婚姻状况_未婚', '婚姻状况_已婚', '婚姻状况_离异', '婚姻状况_丧偶',
            '既往抗抑郁药使用情况_无', '既往抗抑郁药使用情况_使用过抗抑郁药', '既往抗抑郁药使用情况_其它',
            '抑郁程度_轻度', '抑郁程度_中度', '抑郁程度_重度'
        ]
        df = df.iloc[2:].reset_index(drop=True)
    else:
        df = df.iloc[:, :34]
        column_names = [
            '序号', '组别', '随访时的主诉情况_失访',
            '随访时的主诉情况_1_有自杀倾向', '随访时的主诉情况_1_副作用导致停药',
            '随访时的主诉情况_1_失眠', '随访时的主诉情况_1_脱发',
            '随访时的主诉情况_1_激素水平异常', '随访时的主诉情况_1_嗜睡',
            '随访时的主诉情况_1_便秘', '随访时的主诉情况_3_有自杀倾向',
            '随访时的主诉情况_3_副作用导致停药', '随访时的主诉情况_3_失眠',
            '随访时的主诉情况_3_脱发', '随访时的主诉情况_3_激素水平异常',
            '随访时的主诉情况_3_嗜睡', '随访时的主诉情况_3_便秘',
            '随访时的主诉情况_6_有自杀倾向', '随访时的主诉情况_6_副作用导致停药',
            '随访时的主诉情况_6_失眠', '随访时的主诉情况_6_脱发',
            '随访时的主诉情况_6_激素水平异常', '随访时的主诉情况_6_嗜睡',
            '随访时的主诉情况_6_便秘', '随访时的主诉情况_12_有自杀倾向',
            '随访时的主诉情况_12_副作用导致停药', '随访时的主诉情况_12_失眠',
            '随访时的主诉情况_12_脱发', '随访时的主诉情况_12_激素水平异常',
            '随访时的主诉情况_12_嗜睡', '随访时的主诉情况_12_便秘',
            '随访时的主诉情况_是否出现不适症状'
        ]
        if df.shape[1] > len(column_names):
            for i in range(len(column_names), df.shape[1]):
                column_names.append(f'未使用列_{i}')
        elif df.shape[1] < len(column_names):
            column_names = column_names[:df.shape[1]]
        df.columns = column_names
        df = df.iloc[3:].reset_index(drop=True)
    
    # 添加医院标识并创建唯一标识
    if hospital_name:
        df['医院'] = hospital_name
    # 转换序号为字符串，避免数字类型导致的合并问题
    df['序号'] = df['序号'].astype(str).str.strip()
    df['唯一标识'] = df['医院'] + '_' + df['序号']
    
    # 去重：同一医院+序号的样本只保留第一条（解决原始数据重复）
    df = df.drop_duplicates(subset=['唯一标识'], keep='first').reset_index(drop=True)
    return df

# 加载数据（确保唯一标识唯一）
df1 = load_and_clean_data(excel_file1, '一院临床受试者及抑郁症的基本数据', is_basic=True, hospital_name='一院')
df2 = load_and_clean_data(excel_file1, '二院临床受试者及抑郁症的基本数据', is_basic=True, hospital_name='二院')
df3 = load_and_clean_data(excel_file2, '一院随访的抗抑郁药物使用后主诉情况', is_basic=False, hospital_name='一院')
df4 = load_and_clean_data(excel_file2, '二院随访的抗抑郁药物使用后主诉情况', is_basic=False, hospital_name='二院')

# 合并同类型数据（去重）
merged_basic = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['唯一标识'], keep='first')
merged_follow = pd.concat([df3, df4], ignore_index=True).drop_duplicates(subset=['唯一标识'], keep='first')

# 合并基本信息与随访数据：左连接保留所有基本信息样本，避免遗漏
merged_all = pd.merge(
    merged_basic, 
    merged_follow, 
    on=['唯一标识', '组别', '医院', '序号'], 
    how='left',  # 左连接：保留所有有基本信息的样本，即使无随访数据
    suffixes=('', '_follow')  # 处理列名重复
)

# 处理随访数据中的缺失值（无随访数据的症状记为0）
symptom_cols = [col for col in merged_all.columns if '随访时的主诉情况_' in col and any(x in col for x in ['1', '3', '6', '12'])]
for col in symptom_cols:
    merged_all[col] = pd.to_numeric(merged_all[col], errors='coerce').fillna(0)

# 计算加权不适症状程度（兼容缺失随访数据的情况）
time_weights = {"1": 0.1, "3": 0.1, "6": 0.4, "12": 0.4}
symptom_weights = {
    "有自杀倾向": 0.434231 , "副作用导致停药": 0.217116 , 
    "失眠":  0.086846, "脱发": 0.062033 , "激素水平异常": 0.086846 ,
    "嗜睡": 0.056464, "便秘": 0.0565464
}
merged_all['加权不适症状程度'] = 0
for time_period in time_weights:
    for symptom in symptom_weights:
        col = f'随访时的主诉情况_{time_period}_{symptom}'
        if col in merged_all.columns:
            merged_all['加权不适症状程度'] += merged_all[col].fillna(0) * time_weights[time_period] * symptom_weights[symptom]

# 优化婚姻状况分类逻辑（确保正确识别"丧偶"）
def merge_categorical_columns(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix)]
    for col in cols:
        # 转换为数值型，空值视为0（未选中）
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # 当所有列为0时，记为"未知"
    max_col = df[cols].idxmax(axis=1)  # 这里直接用pandas的idxmax，返回Series类型
    max_val = df[cols].max(axis=1)
    # 用pandas的where方法替换np.where，保持Series类型
    max_col = max_col.where(max_val != 0, f'{prefix}未知')
    # 直接对Series使用str.replace
    return max_col.str.replace(prefix, '')

merged_all['婚姻状况'] = merge_categorical_columns(merged_all, '婚姻状况_')
merged_all['既往用药史'] = merge_categorical_columns(merged_all, '既往抗抑郁药使用情况_')
merged_all['初始抑郁程度'] = merge_categorical_columns(merged_all, '抑郁程度_')

# ---------------------- 新增：标记并剔除未知数据 ----------------------
# 1. 标记未知数据（创建标记列）
merged_all['婚姻状况_未知标记'] = merged_all['婚姻状况'] == '未知'
merged_all['既往用药史_未知标记'] = merged_all['既往用药史'] == '未知'
merged_all['初始抑郁程度_未知标记'] = merged_all['初始抑郁程度'] == '未知'

# 2. 统计未知数据数量
print("\n===== 未知数据统计 =====")
print(f"婚姻状况未知样本数：{merged_all['婚姻状况_未知标记'].sum()}")
print(f"既往用药史未知样本数：{merged_all['既往用药史_未知标记'].sum()}")
print(f"初始抑郁程度未知样本数：{merged_all['初始抑郁程度_未知标记'].sum()}")

# 3. 剔除任何一列标记为未知的样本（保留所有非未知数据）
merged_clean = merged_all[
    ~merged_all['婚姻状况_未知标记'] & 
    ~merged_all['既往用药史_未知标记'] & 
    ~merged_all['初始抑郁程度_未知标记']
].reset_index(drop=True)

print(f"剔除未知数据后剩余样本量：{len(merged_clean)}（原始样本量：{len(merged_all)}）")

# ---------------------- 后续分析基于清洗后的数据（merged_clean） ----------------------

# 分组统计函数（保持不变）
def calculate_stats(grouped_data):
    stats_data = grouped_data.agg(['mean', 'var', 'count']).reset_index()
    stats_data.columns = ['分组', '平均值', '方差', '样本量']
    return stats_data

def perform_anova(df, group_col, value_col):
    groups = []
    for name, group in df.groupby(group_col):
        groups.append(group[value_col].dropna())
    if len(groups) < 2:
        return np.nan, np.nan
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value

def print_formatted_results(title, stats_data, f_stat, p_value):
    print(f"{title}：")
    print("分组\t平均值\t方差\t样本量")
    for _, row in stats_data.iterrows():
        print(f"{row['分组']}\t{row['平均值']:.3f}\t{row['方差']:.3f}\t{int(row['样本量'])}")
    print(f"F值：{f_stat:.3f}" if not np.isnan(f_stat) else "F值：分组不足，无法计算")
    print(f"P值：{p_value:.3f}" if not np.isnan(p_value) else "P值：分组不足，无法计算")
    print()

# 调试信息：检查清洗后的数据唯一性和样本量
print("\n===== 清洗后数据基本信息 =====")
print(f"清洗后唯一标识重复数：{merged_clean['唯一标识'].duplicated().sum()}")  # 应输出0
print(f"清洗后各组别样本量：\n{merged_clean.groupby('组别')['唯一标识'].count()}")

# 整体与组别统计（基于清洗后的数据merged_clean）
print("\n===== 清洗后数据统计分析 =====")
overall_marital = calculate_stats(merged_clean.groupby('婚姻状况')['加权不适症状程度'])
overall_med = calculate_stats(merged_clean.groupby('既往用药史')['加权不适症状程度'])
overall_depress = calculate_stats(merged_clean.groupby('初始抑郁程度')['加权不适症状程度'])

f_overall_marital, p_overall_marital = perform_anova(merged_clean, '婚姻状况', '加权不适症状程度')
f_overall_med, p_overall_med = perform_anova(merged_clean, '既往用药史', '加权不适症状程度')
f_overall_depress, p_overall_depress = perform_anova(merged_clean, '初始抑郁程度', '加权不适症状程度')

print_formatted_results("整体-不同婚姻状况的统计分析", overall_marital, f_overall_marital, p_overall_marital)
print_formatted_results("整体-不同既往用药史的统计分析", overall_med, f_overall_med, p_overall_med)
print_formatted_results("整体-不同初始抑郁程度的统计分析", overall_depress, f_overall_depress, p_overall_depress)

# 各组别分析（基于清洗后的数据merged_clean）
group_values = sorted(merged_clean['组别'].unique())
for group in group_values:
    group_data = merged_clean[merged_clean['组别'] == group]
    print(f"\n\n===== 组别 {group} 的统计分析 =====")
    
    group_marital = calculate_stats(group_data.groupby('婚姻状况')['加权不适症状程度'])
    group_med = calculate_stats(group_data.groupby('既往用药史')['加权不适症状程度'])
    group_depress = calculate_stats(group_data.groupby('初始抑郁程度')['加权不适症状程度'])

    f_group_marital, p_group_marital = perform_anova(group_data, '婚姻状况', '加权不适症状程度')
    f_group_med, p_group_med = perform_anova(group_data, '既往用药史', '加权不适症状程度')
    f_group_depress, p_group_depress = perform_anova(group_data, '初始抑郁程度', '加权不适症状程度')

    print_formatted_results(f"组别{group}-不同婚姻状况的统计分析", group_marital, f_group_marital, p_group_marital)
    print_formatted_results(f"组别{group}-不同既往用药史的统计分析", group_med, f_group_med, p_group_med)
    print_formatted_results(f"组别{group}-不同初始抑郁程度的统计分析", group_depress, f_group_depress, p_group_depress)
'''
# 丧偶样本分析（基于清洗后的数据）
print("\n\n===== 清洗后：所有丧偶样本及统计 =====")
widowed_samples = merged_clean[merged_clean['婚姻状况'] == '丧偶'].reset_index(drop=True)
print(f"清洗后丧偶样本总数：{len(widowed_samples)}")

widowed_by_hospital = widowed_samples.groupby('医院').agg(样本量=('唯一标识', 'count')).reset_index()
widowed_by_group = widowed_samples.groupby('组别').agg(样本量=('唯一标识', 'count')).reset_index()

print("\n按医院统计：")
print("医院\t样本量")
for _, row in widowed_by_hospital.iterrows():
    print(f"{row['医院']}\t{int(row['样本量'])}")

print("\n按组别统计：")
print("组别\t样本量")
for _, row in widowed_by_group.iterrows():
    print(f"{row['组别']}\t{int(row['样本量'])}")
'''