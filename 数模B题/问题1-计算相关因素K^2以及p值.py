import numpy as np
from scipy import stats

def chi_square_test(observed):
    """
    计算卡方检验值和p值
    
    参数:
    observed (array-like): 观察频数的二维数组
    
    返回:
    float: 卡方检验值
    float: p值
    """
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    return chi2, p

def analyze_hospital_data(hospital_name, data_sets, is_merged=False):
    """
    分析医院数据并输出卡方检验结果
    
    参数:
    hospital_name (str): 医院名称
    data_sets (dict): 包含各种数据的字典
    is_merged (bool): 是否为合并数据
    """
    prefix = "\n合并数据" if is_merged else f"\n{hospital_name}临床受试者及抑郁症的基本数据"
    print(prefix)
    
    for data_name, data in data_sets.items():
        # 打印数据统计
        print(f"\n{data_name}人数统计")
        print(f"组别\t" + "\t".join(data['categories']))
        for i, row in enumerate(data['data']):
            print(f"{i+1}\t" + "\t".join([f"{val:.1f}" for val in row]))
        
        # 计算卡方检验
        chi2, p = chi_square_test(data['data'])
        
        # 输出卡方检验结果
        print(f"卡方检验结果")
        print(f"组别与{data_name}的卡方值：{chi2:.4f}，p 值：{p:.4f}。")
        significance = "存在显著关联" if p < 0.05 else "无显著关联"
        print(f"p 值{'小于' if p < 0.05 else '大于'}0.05，{'说明' if p < 0.05 else '意味着'}在{hospital_name}数据中，组别与{data_name}之间{significance}。")

def merge_hospital_data(data_set1, data_set2):
    """
    合并两个医院的数据集
    
    参数:
    data_set1 (dict): 第一个医院的数据集
    data_set2 (dict): 第二个医院的数据集
    
    返回:
    dict: 合并后的数据集
    """
    merged_data = {}
    
    for data_name in data_set1.keys():
        if data_name in data_set2:
            # 确保两个数据集的类别相同
            if data_set1[data_name]['categories'] == data_set2[data_name]['categories']:
                merged_data[data_name] = {
                    'categories': data_set1[data_name]['categories'],
                    'data': data_set1[data_name]['data'] + data_set2[data_name]['data']
                }
            else:
                print(f"警告: {data_name}的类别不匹配，无法合并")
    
    return merged_data

# 一院数据
hospital1_data = {
    "婚姻状况": {
        'categories': ["婚姻状况_未婚", "婚姻状况_已婚", "婚姻状况_离异", "婚姻状况_丧偶"],
        'data': np.array([
            [222.0, 237.0, 48.0, 20.0],
            [221.0, 244.0, 38.0, 22.0],
            [204.0, 170.0, 49.0, 34.0]
        ])
    },
    "既往抗抑郁药使用情况": {
        'categories': ["既往抗抑郁药使用情况_无", "既往抗抑郁药使用情况_使用过抗抑郁药", "既往抗抑郁药使用情况_其它"],
        'data': np.array([
            [231.0, 102.0, 192.0],
            [235.0, 97.0, 193.0],
            [252.0, 102.0, 171.0]
        ])
    },
    "抑郁程度": {
        'categories': ["抑郁程度_轻度", "抑郁程度_中度", "抑郁程度_重度"],
        'data': np.array([
            [41.0, 433.0, 51.0],
            [58.0, 417.0, 50.0],
            [62.0, 426.0, 37.0]
        ])
    }
}

# 二院数据
hospital2_data = {
    "婚姻状况": {
        'categories': ["婚姻状况_未婚", "婚姻状况_已婚", "婚姻状况_离异", "婚姻状况_丧偶"],
        'data': np.array([
            [149.0, 325.0, 46.0, 4.0],
            [155.0, 323.0, 36.0, 11.0],
            [141.0, 333.0, 45.0, 4.0]
        ])
    },
    "既往抗抑郁药使用情况": {
        'categories': ["既往抗抑郁药使用情况_无", "既往抗抑郁药使用情况_使用过抗抑郁药", "既往抗抑郁药使用情况_其它"],
        'data': np.array([
            [282.0, 27.0, 215.0],
            [260.0, 32.0, 233.0],
            [257.0, 29.0, 240.0]
        ])
    },
    "抑郁程度": {
        'categories': ["抑郁程度_轻度", "抑郁程度_中度", "抑郁程度_重度"],
        'data': np.array([
            [44.0, 429.0, 39.0],
            [31.0, 433.0, 41.0],
            [26.0, 443.0, 46.0]
        ])
    }
}

# 合并数据
merged_data = merge_hospital_data(hospital1_data, hospital2_data)

# 分析一院数据
analyze_hospital_data("一院", hospital1_data)

# 分析二院数据
analyze_hospital_data("二院", hospital2_data)

# 分析合并数据
analyze_hospital_data("一院和二院", merged_data, is_merged=True)