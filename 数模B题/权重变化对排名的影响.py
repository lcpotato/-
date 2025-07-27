import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
import random

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 熵权法计算权重
def entropy_weight(data):
    norm_df = data / np.sqrt(np.sum(data**2, axis=0))
    p = norm_df / norm_df.sum(axis=0)
    p = p.replace(0, 1e-10)  # 避免log(0)
    e = -1 / np.log(len(p)) * np.sum(p * np.log(p), axis=0)
    d = 1 - e  # 差异系数
    weights = d / np.sum(d)  # 权重
    return weights.tolist()

# TOPSIS分析函数
def topsis_analysis(data, weights, criteria_types):
    norm_df = data / np.sqrt(np.sum(data**2, axis=0))
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
    
    # 计算距离和相对接近度
    s_best = np.sqrt(np.sum((weighted_norm_df - ideal_best)**2, axis=1))
    s_worst = np.sqrt(np.sum((weighted_norm_df - ideal_worst)**2, axis=1))
    performance_score = s_worst / (s_best + s_worst)
    
    result_df = pd.DataFrame(index=data.index)
    result_df['理想解距离'] = s_best
    result_df['负理想解距离'] = s_worst
    result_df['相对接近度'] = performance_score
    result_df['排名'] = result_df['相对接近度'].rank(ascending=False)
    
    return result_df

# 生成权重扰动
def generate_weight_variations(base_weight, range_pct=0.15, step=0.005):
    lower_bound = max(0, base_weight - base_weight * range_pct)
    upper_bound = min(1, base_weight + base_weight * range_pct)
    return [round(w, 3) for w in np.arange(lower_bound, upper_bound + step, step) if lower_bound <= w <= upper_bound]

# 生成发生率扰动
def generate_incidence_variations(base_value, range_pct=0.15, step=0.05):
    lower_bound = max(0, base_value - base_value * range_pct)
    upper_bound = base_value + base_value * range_pct
    return [round(w, 3) for w in np.arange(lower_bound, upper_bound + step, step) if lower_bound <= w <= upper_bound]

# 主函数：生成可视化报告
def generate_visual_report():
    # 1. 基础数据设置
    base_data = {
        '组别': [1, 2, 3],
        '第一月不良症状发生率': [1.69, 1.60, 1.56],
        '第三月不良症状发生率': [0.92, 0.86, 0.76],
        '第六月不良症状发生率': [0.51, 0.48, 0.43],
        '第十二月不良症状发生率': [0.28, 0.27, 0.20],
        '总体恢复率': [0.91, 0.88, 0.91],
        '复发率': [0.25, 0.35, 0.20],
        '改善指数': [0.6756, 0.6712, 0.7341]  
    }
    
    criteria_types = [
        '成本型', '成本型', '成本型', '成本型',  # 各月发生率
        '效益型', '成本型', '效益型'  # 恢复率、复发率、改善指数
    ]
    
    # 2. 输出目录设置
    output_dir = r"C:\Users\22805\Desktop\数模B题"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "抗抑郁药综合敏感度分析可视化.pdf")
    
    # 3. 基础数据分析
    df_base = pd.DataFrame(base_data).set_index('组别')
    weights_base = entropy_weight(df_base)
    result_base = topsis_analysis(df_base, weights_base, criteria_types)
    
    # 4. 生成扰动数据
    # 权重扰动组合
    omega3_values = generate_weight_variations(0.2)
    omega6_values = generate_weight_variations(0.3)
    omega12_values = generate_weight_variations(0.5)
    
    # 发生率扰动组合
    incidence_cols = [
        '第一月不良症状发生率', 
        '第三月不良症状发生率', 
        '第六月不良症状发生率', 
        '第十二月不良症状发生率'
    ]
    
    incidence_variations = {}
    for group in [1, 2, 3]:
        incidence_variations[group] = {
            col: generate_incidence_variations(df_base.loc[group, col]) 
            for col in incidence_cols
        }
    
    # 5. 敏感度分析（限制组合数量避免计算过载）
    MAX_COMBINATIONS = 500
    weight_combos = list(product(omega3_values, omega6_values, omega12_values))
    sampled_weights = random.sample(weight_combos, min(MAX_COMBINATIONS, len(weight_combos)))
    
    sensitivity_results = []
    for w3, w6, w12 in sampled_weights:
        # 随机选择发生率扰动值
        incidence_vals = {}
        for group in [1, 2, 3]:
            incidence_vals[group] = {
                col: random.choice(incidence_variations[group][col]) 
                for col in incidence_cols
            }
        
        # 计算新的改善指数
        df_perturb = df_base.copy()
        for group in [1, 2, 3]:
            x1 = df_perturb.loc[group, '第一月不良症状发生率']
            x3 = df_perturb.loc[group, '第三月不良症状发生率']
            x6 = df_perturb.loc[group, '第六月不良症状发生率']
            x12 = df_perturb.loc[group, '第十二月不良症状发生率']
            
            if x1 != 0:
                improve_3 = (x1 - x3) / x1
                improve_6 = (x1 - x6) / x1
                improve_12 = (x1 - x12) / x1
                df_perturb.loc[group, '改善指数'] = round(w3*improve_3 + w6*improve_6 + w12*improve_12, 4)
        
        # 计算TOPSIS结果和排名变化
        result_perturb = topsis_analysis(df_perturb, weights_base, criteria_types)
        rank_changes = {
            group: abs(result_perturb.loc[group, '排名'] - result_base.loc[group, '排名'])
            for group in [1, 2, 3]
        }
        
        sensitivity_results.append({
            'omega3': w3,
            'omega6': w6,
            'omega12': w12,
            'incidence_values': incidence_vals,
            'rank_changes': rank_changes,
            'max_rank_change': max(rank_changes.values())
        })
    
    # 6. 生成PDF报告并直接显示图像
    with PdfPages(pdf_path) as pdf:
        # 基础结果可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 权重分布
        axes[0].bar(df_base.columns, weights_base)
        axes[0].set_title('熵权法计算的指标权重')
        axes[0].set_ylabel('权重')
        axes[0].tick_params(axis='x', rotation=90)
        
        # TOPSIS结果
        axes[1].bar([f'组别{i}' for i in [1, 2, 3]], result_base['相对接近度'])
        axes[1].set_title('TOPSIS分析结果（相对接近度）')
        axes[1].set_ylabel('相对接近度')
        axes[1].set_ylim(0, 1)
        
        for i, v in enumerate(result_base['相对接近度']):
            axes[1].text(i, v + 0.02, f"排名: {int(result_base.iloc[i]['排名'])}", ha='center')
        
        plt.tight_layout()
        pdf.savefig()  # 保存到PDF
        plt.show()     # 直接显示图像
        plt.close()
        
        # 敏感度分析结果可视化
        if sensitivity_results:
            # 提取关键数据
            x_vals = [res['omega3'] for res in sensitivity_results]
            y_vals = [res['omega6'] for res in sensitivity_results]
            z_vals = [res['omega12'] for res in sensitivity_results]
            rank_changes = [res['max_rank_change'] for res in sensitivity_results]
            
            # 排名变化热力图
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(x_vals, y_vals, z_vals, c=rank_changes, 
                                cmap='viridis', s=50, alpha=0.7)
            
            ax.set_xlabel('omega3权重')
            ax.set_ylabel('omega6权重')
            ax.set_zlabel('omega12权重')
            ax.set_title('权重组合对排名变化的影响')
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('最大排名变化')
            
            plt.tight_layout()
            pdf.savefig()  # 保存到PDF
            plt.show()     # 直接显示图像
            plt.close()
            
            # 发生率敏感度分析
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(incidence_cols):
                # 计算每个发生率的平均排名影响
                inc_impacts = []
                for group in [1, 2, 3]:
                    base_val = df_base.loc[group, col]
                    for res in sensitivity_results:
                        new_val = res['incidence_values'][group][col]
                        change_pct = (new_val / base_val - 1) * 100
                        avg_rank_change = sum(res['rank_changes'][g] for g in [1, 2, 3]) / 3
                        inc_impacts.append((change_pct, avg_rank_change))
                
                # 绘制散点图
                x, y = zip(*inc_impacts) if inc_impacts else ([], [])
                axes[i].scatter(x, y, alpha=0.5)
                axes[i].set_title(f'{col}变化对排名的影响')
                axes[i].set_xlabel('发生率变化百分比 (%)')
                axes[i].set_ylabel('平均排名变化')
            
            plt.tight_layout()
            pdf.savefig()  # 保存到PDF
            plt.show()     # 直接显示图像
            plt.close()
    
    print(f"可视化报告已生成: {pdf_path}")

if __name__ == "__main__":
    generate_visual_report()