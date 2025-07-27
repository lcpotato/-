import pandas as pd

# 为附件2的工作表设置列名
new_header_2 = ['序号', '组别', '随访时的主诉情况_失访', '随访时的主诉情况_有自杀倾向', '随访时的主诉情况_副作用导致停药', '随访时的主诉情况_失眠',
                '随访时的主诉情况_脱发', '随访时的主诉情况_激素水平异常', '随访时的主诉情况_嗜睡', '随访时的主诉情况_其它',
                '随访时的主诉情况_性功能障碍', '随访时的主诉情况_认知障碍', '随访时的主诉情况_疲劳', '随访时的主诉情况_体重增加', '随访时的主诉情况_口干',
                '随访时的主诉情况_恶心呕吐', '随访时的主诉情况_激越', '随访时的主诉情况_头晕', '随访时的主诉情况_头痛', '随访时的主诉情况_视物模糊',
                '随访时的主诉情况_震颤', '随访时的主诉情况_出汗', '随访时的主诉情况_腹泻', '随访时的主诉情况_心悸', '随访时的主诉情况_皮疹',
                '随访时的主诉情况_呼吸困难', '随访时的主诉情况_低血压', '随访时的主诉情况_关节肌肉疼痛', '随访时的主诉情况_排尿困难',
                '随访时的主诉情况_光敏反应', '随访时的主诉情况_耳鸣', '随访时的主诉情况_口腔炎', '随访时的主诉情况_肝功能异常',
                '随访时的主诉情况_是否出现不适症状', '随访时的主诉情况_是否出现不适症状_第一月', '随访时的主诉情况_是否出现不适症状_第三月',
                '随访时的主诉情况_是否出现不适症状_第六月', '随访时的主诉情况_是否出现不适症状_第十二月']

# 读取附件2
excel_file2 = pd.ExcelFile(r"C:\Users\22805\Desktop\附件2：两个医院随访的抗抑郁药使用后主诉情况.xlsx")

# 读取一院和二院数据并合并
df_one = excel_file2.parse('一院随访的抗抑郁药物使用后主诉情况', header=2)
df_one.columns = new_header_2
df_two = excel_file2.parse('二院随访的抗抑郁药物使用后主诉情况', header=2)
df_two.columns = new_header_2
df = pd.concat([df_one, df_two], ignore_index=True)


# ---------------------- 关键修改：发生率计算（增加二值化处理） ----------------------
# 定义需要计算发生率的列
incidence_cols = [
    '随访时的主诉情况_是否出现不适症状_第一月',
    '随访时的主诉情况_是否出现不适症状_第三月',
    '随访时的主诉情况_是否出现不适症状_第六月',
    '随访时的主诉情况_是否出现不适症状_第十二月'
]

# 二值化处理：将"出现次数>0"视为"出现（1）"，"0次"视为"未出现（0）"
# （解决原始数据可能为次数而非0/1的问题，确保发生率是"出现人数占比"）
for col in incidence_cols:
    df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

# 按组别计算各月发生率（均值即发生率，保留2位小数）
incidence_rate = df.groupby('组别')[incidence_cols].mean().round(2)


# 计算复发率（逻辑不变，基于二值化后的结果更准确）
def calculate_relapse_rate(group):
    relapse_count = 0
    total_count = 0
    for i in range(len(group)):
        first_month = group.iloc[i]['随访时的主诉情况_是否出现不适症状_第一月']
        third_month = group.iloc[i]['随访时的主诉情况_是否出现不适症状_第三月']
        sixth_month = group.iloc[i]['随访时的主诉情况_是否出现不适症状_第六月']
        twelfth_month = group.iloc[i]['随访时的主诉情况_是否出现不适症状_第十二月']

        # 仅统计曾出现过症状的个体
        if first_month == 1 or third_month == 1 or sixth_month == 1:
            total_count += 1
            # 判定复发：前序出现过，后续再次出现
            if (first_month == 1 and (third_month == 1 or sixth_month == 1 or twelfth_month == 1)) or \
               (third_month == 1 and (sixth_month == 1 or twelfth_month == 1)) or \
               (sixth_month == 1 and twelfth_month == 1):
                relapse_count += 1
    return relapse_count / total_count if total_count > 0 else 0


# 计算总体恢复率（逻辑不变，依赖二值化后的数据）
symptom_columns = [col for col in df.columns if '随访时的主诉情况' in col and '是否出现不适症状' not in col]
def calculate_overall_recovery_rate(group):
    recovery_count = 0
    total_count = 0
    for _, row in group.iterrows():
        ever_had_symptom = False
        all_recovered = True
        # 检查是否曾出现过具体症状
        for col in symptom_columns:
            if pd.notnull(row[col]) and row[col] == 1:
                ever_had_symptom = True
                # 基于二值化后的月份数据判断是否完全恢复
                first = row['随访时的主诉情况_是否出现不适症状_第一月']
                third = row['随访时的主诉情况_是否出现不适症状_第三月']
                sixth = row['随访时的主诉情况_是否出现不适症状_第六月']
                twelfth = row['随访时的主诉情况_是否出现不适症状_第十二月']
                if (first == 1 and (third == 1 or sixth == 1 or twelfth == 1)) or \
                   (third == 1 and (sixth == 1 or twelfth == 1)) or \
                   (sixth == 1 and twelfth == 1):
                    all_recovered = False
                    break
        if ever_had_symptom:
            total_count += 1
            if all_recovered:
                recovery_count += 1
    return recovery_count / total_count if total_count > 0 else 0


# 计算复发率和恢复率
relapse_rate = df.groupby('组别').apply(calculate_relapse_rate).round(2)
overall_recovery_rate = df.groupby('组别').apply(calculate_overall_recovery_rate).round(2)

# 整理结果
result = pd.DataFrame({
    '第一月不适症状发生率': incidence_rate['随访时的主诉情况_是否出现不适症状_第一月'],
    '第三月不适症状发生率': incidence_rate['随访时的主诉情况_是否出现不适症状_第三月'],
    '第六月不适症状发生率': incidence_rate['随访时的主诉情况_是否出现不适症状_第六月'],
    '第十二月不适症状发生率': incidence_rate['随访时的主诉情况_是否出现不适症状_第十二月'],
    '总体恢复率': overall_recovery_rate,
    '复发率': relapse_rate
})

print('三种抗抑郁药物的疗效评估结果：')
print(result)