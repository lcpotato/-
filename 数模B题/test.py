import pandas as pd
import os
from itertools import product

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

# 配置文件路径
excel_path = r"C:\Users\22805\Desktop\附件2：两个医院随访的抗抑郁药使用后主诉情况.xlsx"
output_dir = r"C:\Users\22805\Desktop\数模B题"
output_path = os.path.join(output_dir, "抗抑郁药副作用分析_权重敏感度分析_0.005精度.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取Excel文件
try:
    excel_file = pd.ExcelFile(excel_path)
    df_one = excel_file.parse('一院随访的抗抑郁药物使用后主诉情况', header=2)
    df_one.columns = new_header_2
    df_two = excel_file.parse('二院随访的抗抑郁药物使用后主诉情况', header=2)
    df_two.columns = new_header_2
    merged_df = pd.concat([df_one, df_two], ignore_index=True)
except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    exit(1)

# ---------------------- 2. 验证并修正数据（关键步骤） ----------------------
incidence_cols = [
    '随访时的主诉情况_是否出现不适症状_第一月',
    '随访时的主诉情况_是否出现不适症状_第三月',
    '随访时的主诉情况_是否出现不适症状_第六月',
    '随访时的主诉情况_是否出现不适症状_第十二月'
]

# 修正逻辑：将“次数>0”转为1（出现），0保持0（未出现）
for col in incidence_cols:
    merged_df[col] = merged_df[col].apply(lambda x: 1 if x > 0 else 0)

# ---------------------- 3. 计算发生率（修正后，范围0~1） ----------------------
incidence_rate = merged_df.groupby('组别')[incidence_cols].mean().round(4)  # 保留4位更精确
incidence_rate.columns = ['1月发生率', '3月发生率', '6月发生率', '12月发生率']

# ---------------------- 4. 权重敏感度分析 ----------------------
# 基础权重
base_weights = {'omega3': 0.2, 'omega6': 0.3, 'omega12': 0.5}

# 生成权重变化范围（±15%，步长0.005）
def generate_weight_variations(base_weight):
    lower_bound = max(0, base_weight - 0.15)
    upper_bound = min(1, base_weight + 0.15)
    return [round(base_weight + i * 0.005, 3) 
            for i in range(int(-0.15/0.005), int(0.15/0.005) + 1)
            if lower_bound <= base_weight + i * 0.005 <= upper_bound]

omega3_values = generate_weight_variations(base_weights['omega3'])
omega6_values = generate_weight_variations(base_weights['omega6'])
omega12_values = generate_weight_variations(base_weights['omega12'])

# 确保权重和为1（精度0.005）
valid_combinations = []
for w3, w6, w12 in product(omega3_values, omega6_values, omega12_values):
    if abs(w3 + w6 + w12 - 1) < 0.0025:  # 考虑浮点数精度问题
        valid_combinations.append((w3, w6, w12))

# 限制组合数量，避免生成过多结果
MAX_COMBINATIONS = 2000
if len(valid_combinations) > MAX_COMBINATIONS:
    print(f"警告: 找到 {len(valid_combinations)} 个有效权重组合，将只分析前 {MAX_COMBINATIONS} 个组合以控制文件大小")
    valid_combinations = valid_combinations[:MAX_COMBINATIONS]

# ---------------------- 5. 计算不同权重下的改善指数并输出结果 ----------------------
try:
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入分析配置信息
        f.write("=== 抗抑郁药副作用改善指数权重敏感度分析 ===\n")
        f.write(f"基础权重: omega3={base_weights['omega3']:.3f}, omega6={base_weights['omega6']:.3f}, omega12={base_weights['omega12']:.3f}\n")
        f.write(f"权重变化范围: 基础值 ±15%\n")
        f.write(f"权重精度: 0.005\n")
        f.write(f"权重和验证精度: 0.005\n\n")
        
        # 基础改善指数计算
        def calculate_improvement_index(row, omega3, omega6, omega12):
            x1 = row['1月发生率']
            x3 = row['3月发生率']
            x6 = row['6月发生率']
            x12 = row['12月发生率']
            
            if x1 == 0:
                return 0.0
            
            improve_3 = (x1 - x3) / x1 if x1 != 0 else 0
            improve_6 = (x1 - x6) / x1 if x1 != 0 else 0
            improve_12 = (x1 - x12) / x1 if x1 != 0 else 0
            
            index = omega3 * improve_3 + omega6 * improve_6 + omega12 * improve_12
            return round(index, 4)
        
        # 计算基础改善指数
        base_result = incidence_rate.copy()
        base_result['改善指数'] = base_result.apply(
            lambda row: calculate_improvement_index(row, base_weights['omega3'], base_weights['omega6'], base_weights['omega12']), 
            axis=1
        )
        
        f.write("=== 基础权重下的分析结果 ===\n")
        f.write(base_result.to_string())
        f.write("\n\n")
        
        # 写入敏感度分析结果
        f.write(f"=== 权重敏感度分析结果 ===\n")
        f.write(f"共找到 {len(valid_combinations)} 个满足条件的权重组合，展示前 {len(valid_combinations)} 个:\n\n")
        
        # 为结果添加排序（按改善指数变化幅度）
        base_improvement = base_result['改善指数'].to_dict()
        result_combinations = []
        
        for i, (w3, w6, w12) in enumerate(valid_combinations, 1):
            current_result = incidence_rate.copy()
            current_result['改善指数'] = current_result.apply(
                lambda row: calculate_improvement_index(row, w3, w6, w12), 
                axis=1
            )
            
            # 计算与基础结果的差异
            max_diff = max(abs(current_result.loc[group, '改善指数'] - base_improvement[group]) 
                          for group in current_result.index)
            
            result_combinations.append((w3, w6, w12, current_result, max_diff))
        
        # 按最大差异排序
        result_combinations.sort(key=lambda x: x[4], reverse=True)
        
        # 输出排序后的结果
        for i, (w3, w6, w12, current_result, max_diff) in enumerate(result_combinations, 1):
            f.write(f"组合 {i} (差异排名: {i}): omega3={w3:.3f}, omega6={w6:.3f}, omega12={w12:.3f}\n")
            f.write(f"与基础结果的最大差异: {max_diff:.4f}\n")
            f.write(current_result.to_string())
            f.write("\n\n")
    
    print(f"分析结果已成功保存至: {output_path}")
    print(f"共分析 {len(valid_combinations)} 组权重组合")
    
except Exception as e:
    print(f"写入结果文件时出错: {e}")