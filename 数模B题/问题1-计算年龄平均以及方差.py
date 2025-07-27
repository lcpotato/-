import pandas as pd
from scipy import stats

# 读取文件
try:
    excel_file = pd.ExcelFile(r'C:\Users\22805\Desktop\附件1：两个院临床受试者及抑郁症的基本数据.xlsx')
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

# 获取所有表名
sheet_names = excel_file.sheet_names
if len(sheet_names) < 2:
    print("文件中至少应有两个工作表")
    exit()

# 读取一院和二院的数据
try:
    df_hospital1 = excel_file.parse(sheet_names[0])
    df_hospital2 = excel_file.parse(sheet_names[1])
except Exception as e:
    print(f"读取工作表时出错: {e}")
    exit()

# 查找组别列和年龄列的函数
def find_columns(df):
    group_col = None
    age_col = None

    for col in df.columns:
        if '组别' in col:
            group_col = col
        if '年龄' in col and '岁' in col:
            age_col = col

    # 验证是否找到所需列
    if group_col is None:
        print("未找到组别列")
        return None, None
    if age_col is None:
        print("未找到年龄（岁）列")
        return None, None
        
    return group_col, age_col

# 处理一院数据
group_col1, age_col1 = find_columns(df_hospital1)
if group_col1 and age_col1:
    # 确保组别列和年龄列的数据类型正确
    df_hospital1[group_col1] = pd.to_numeric(df_hospital1[group_col1], errors='coerce')
    df_hospital1[age_col1] = pd.to_numeric(df_hospital1[age_col1], errors='coerce')
    
    # 筛选出组别为1、2、3的数据
    valid_groups1 = df_hospital1[df_hospital1[group_col1].isin([1, 2, 3])]
    
    # 按组别计算年龄的均值和方差
    stats_hospital1 = valid_groups1.groupby(group_col1)[age_col1].agg(['mean', 'var']).reset_index()
    
    # 重命名列以去除英文
    stats_hospital1.columns = ['组别', '平均年龄', '年龄方差']
    
    # 格式化结果
    stats_hospital1['平均年龄'] = stats_hospital1['平均年龄'].map('{:.2f}'.format)
    stats_hospital1['年龄方差'] = stats_hospital1['年龄方差'].map('{:.2f}'.format)
    
    # 输出结果
    print(f"\n{sheet_names[0]}各年龄段的统计数据")
    print("-" * 40)
    print("| 组别 | 平均年龄 | 年龄方差 |")
    print("|------|----------|----------|")
    for _, row in stats_hospital1.iterrows():
        print(f"| {row['组别']}    | {row['平均年龄']}     | {row['年龄方差']}     |")
    print("-" * 40)
    
    # 方差分析
    group1 = valid_groups1[valid_groups1[group_col1] == 1][age_col1].dropna()
    group2 = valid_groups1[valid_groups1[group_col1] == 2][age_col1].dropna()
    group3 = valid_groups1[valid_groups1[group_col1] == 3][age_col1].dropna()
    
    f_value, p_value = stats.f_oneway(group1, group2, group3)
    
    print(f"\n{sheet_names[0]}方差分析结果:")
    print(f"F值: {f_value:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print("结论: 各组之间年龄存在显著差异 (p < 0.05)")
    else:
        print("结论: 各组之间年龄不存在显著差异 (p >= 0.05)")

# 处理二院数据
group_col2, age_col2 = find_columns(df_hospital2)
if group_col2 and age_col2:
    # 确保组别列和年龄列的数据类型正确
    df_hospital2[group_col2] = pd.to_numeric(df_hospital2[group_col2], errors='coerce')
    df_hospital2[age_col2] = pd.to_numeric(df_hospital2[age_col2], errors='coerce')
    
    # 筛选出组别为1、2、3的数据
    valid_groups2 = df_hospital2[df_hospital2[group_col2].isin([1, 2, 3])]
    
    # 按组别计算年龄的均值和方差
    stats_hospital2 = valid_groups2.groupby(group_col2)[age_col2].agg(['mean', 'var']).reset_index()
    
    # 重命名列以去除英文
    stats_hospital2.columns = ['组别', '平均年龄', '年龄方差']
    
    # 格式化结果
    stats_hospital2['平均年龄'] = stats_hospital2['平均年龄'].map('{:.2f}'.format)
    stats_hospital2['年龄方差'] = stats_hospital2['年龄方差'].map('{:.2f}'.format)
    
    # 输出结果
    print(f"\n{sheet_names[1]}各年龄段的统计数据")
    print("-" * 40)
    print("| 组别 | 平均年龄 | 年龄方差 |")
    print("|------|----------|----------|")
    for _, row in stats_hospital2.iterrows():
        print(f"| {row['组别']}    | {row['平均年龄']}     | {row['年龄方差']}     |")
    print("-" * 40)
    
    # 方差分析
    group1 = valid_groups2[valid_groups2[group_col2] == 1][age_col2].dropna()
    group2 = valid_groups2[valid_groups2[group_col2] == 2][age_col2].dropna()
    group3 = valid_groups2[valid_groups2[group_col2] == 3][age_col2].dropna()
    
    f_value, p_value = stats.f_oneway(group1, group2, group3)
    
    print(f"\n{sheet_names[1]}方差分析结果:")
    print(f"F值: {f_value:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print("结论: 各组之间年龄存在显著差异 (p < 0.05)")
    else:
        print("结论: 各组之间年龄不存在显著差异 (p >= 0.05)")

# 汇总数据
if group_col1 and age_col1 and group_col2 and age_col2:
    # 合并有效数据
    combined_data = pd.concat([valid_groups1[[group_col1, age_col1]], 
                              valid_groups2[[group_col2, age_col2]]], 
                             ignore_index=True)
    combined_data.columns = ['组别', '年龄']
    
    # 按组别计算年龄的均值和方差
    stats_total = combined_data.groupby('组别')['年龄'].agg(['mean', 'var']).reset_index()
    
    # 重命名列以去除英文
    stats_total.columns = ['组别', '平均年龄', '年龄方差']
    
    # 格式化结果
    stats_total['平均年龄'] = stats_total['平均年龄'].map('{:.2f}'.format)
    stats_total['年龄方差'] = stats_total['年龄方差'].map('{:.2f}'.format)
    
    # 输出结果
    print("\n汇总各年龄段的统计数据")
    print("-" * 40)
    print("| 组别 | 平均年龄 | 年龄方差 |")
    print("|------|----------|----------|")
    for _, row in stats_total.iterrows():
        print(f"| {row['组别']}    | {row['平均年龄']}     | {row['年龄方差']}     |")
    print("-" * 40)
    
    # 汇总数据的方差分析
    group1 = combined_data[combined_data['组别'] == 1]['年龄'].dropna()
    group2 = combined_data[combined_data['组别'] == 2]['年龄'].dropna()
    group3 = combined_data[combined_data['组别'] == 3]['年龄'].dropna()
    
    f_value, p_value = stats.f_oneway(group1, group2, group3)
    
    print(f"\n汇总数据方差分析结果:")
    print(f"F值: {f_value:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print("结论: 各组之间年龄存在显著差异 (p < 0.05)")
    else:
        print("结论: 各组之间年龄不存在显著差异 (p >= 0.05)")