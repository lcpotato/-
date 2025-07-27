import pandas as pd

# ---------------------- 1. 数据读取与预处理 ----------------------
new_header_2 = [
    '序号', '组别', '随访时的主诉情况_失访', '随访时的主诉情况_有自杀倾向',
    '随访时的主诉情况_副作用导致停药', '随访时的主诉情况_失眠', '随访时的主诉情况_脱发',
    '随访时的主诉情况_激素水平异常', '随访时的主诉情况_嗜睡', '随访时的主诉情况_其它',
    '随访时的主诉情况_性功能障碍', '随访时的主诉情况_认知障碍', '随访时的主诉情况_疲劳',
    '随访时的主诉情况_体重增加', '随访时的主诉情况_口干', '随访时的主诉情况_恶心呕吐',
    '随访时的主诉情况_激越', '随访时的主诉情况_头晕', '随访时的主诉情况_头痛',
    '随访时的主诉情况_视物模糊', '随访时的主诉情况_震颤', '随访时的主诉情况_出汗',
    '随访时的主诉情况_腹泻', '随访时的主诉情况_心悸', '随访时的主诉情况_皮疹',
    '随访时的主诉情况_呼吸困难', '随访时的主诉情况_低血压', '随访时的主诉情况_关节肌肉疼痛',
    '随访时的主诉情况_排尿困难', '随访时的主诉情况_光敏反应', '随访时的主诉情况_耳鸣',
    '随访时的主诉情况_口腔炎', '随访时的主诉情况_肝功能异常', '随访时的主诉情况_是否出现不适症状',
    '随访时的主诉情况_是否出现不适症状_第一月', '随访时的主诉情况_是否出现不适症状_第三月',
    '随访时的主诉情况_是否出现不适症状_第六月', '随访时的主诉情况_是否出现不适症状_第十二月'
]

excel_path = r"C:\Users\22805\Desktop\附件2：两个医院随访的抗抑郁药使用后主诉情况.xlsx"
excel_file = pd.ExcelFile(excel_path)

df_one = excel_file.parse('一院随访的抗抑郁药物使用后主诉情况', header=2)
df_one.columns = new_header_2
df_two = excel_file.parse('二院随访的抗抑郁药物使用后主诉情况', header=2)
df_two.columns = new_header_2
merged_df = pd.concat([df_one, df_two], ignore_index=True)


# ---------------------- 2. 验证并修正数据（关键步骤） ----------------------
incidence_cols = [
    '随访时的主诉情况_是否出现不适症状_第一月',
    '随访时的主诉情况_是否出现不适症状_第三月',
    '随访时的主诉情况_是否出现不适症状_第六月',
    '随访时的主诉情况_是否出现不适症状_第十二月'
]

# 检查原始数据分布（调试用，可注释）
for col in incidence_cols:
    print(f"\n=== 调试：{col} 的原始分布 ===")
    print(merged_df[col].value_counts(dropna=False))
    print(merged_df[col].describe())

# 修正逻辑：将“次数>0”转为1（出现），0保持0（未出现）
for col in incidence_cols:
    merged_df[col] = merged_df[col].apply(lambda x: 1 if x > 0 else 0)


# ---------------------- 3. 计算发生率（修正后，范围0~1） ----------------------
incidence_rate = merged_df.groupby('组别')[incidence_cols].mean().round(4)  # 保留4位更精确
incidence_rate.columns = ['1月发生率', '3月发生率', '6月发生率', '12月发生率']


# ---------------------- 4. 计算改善指数 ----------------------
def calculate_improvement_index(row):
    x1 = row['1月发生率']
    x3 = row['3月发生率']
    x6 = row['6月发生率']
    x12 = row['12月发生率']
    
    if x1 == 0:
        return 0.0
    
    improve_3 = (x1 - x3) / x1 if x1 != 0 else 0
    improve_6 = (x1 - x6) / x1 if x1 != 0 else 0
    improve_12 = (x1 - x12) / x1 if x1 != 0 else 0
    
    omega3, omega6, omega12 = 0.2, 0.3, 0.5
    index = omega3 * improve_3 + omega6 * improve_6 + omega12 * improve_12
    
    return round(index, 4)

incidence_rate['改善指数'] = incidence_rate.apply(calculate_improvement_index, axis=1)


# ---------------------- 5. 输出结果 ----------------------
print("=== 修正后：各分组不适症状发生率及改善指数 ===")
print(incidence_rate)