import numpy as np
from scipy.stats import qmc
import pandas as pd
import os
import time

# ==========================================
# 1. 定义变量范围 (已同步最新防作弊物理约束)
# ==========================================
bounds = {
    'L1': [3.5, 12.5],  # mm: 臂长
    'r2': [2.0, 7],  # mm: 环外径
    't1': [-60, 0],  # deg: 角度1
    't2': [0, 60.0],  # deg: 角度2
    'w3': [0.25, 0.8],  # mm: 馈电线宽
    'w4': [0.2, 0.8]  # mm: 环宽
}
# 物理安全限制 (用于内部筛选，不输出)
MIN_R3_LIMIT = 0.5

# 🔥 最终输出列 (不包含 r3)
columns_export = ['L1', 'r2', 'w4', 't1', 't2', 'w3']

# ==========================================
# 2. 生成 & 过滤 (Rejection Sampling)
# ==========================================
target_samples = 1000
n_vars = 6

print(f"🚀 开始生成真随机数据 (目标: {target_samples} 组)...")
print(f"   👉 导出格式: 单个 TXT 文件 (Tab分隔), 无r3列")

valid_data = pd.DataFrame()
batch_count = 0

# 循环直到攒够数据
while len(valid_data) < target_samples:
    batch_count += 1

    # 💥 核心修复：利用当前毫秒级时间戳作为真随机种子，彻底打破重复魔咒！
    current_seed = int(time.time() * 1000) % (2 ** 32)
    sampler = qmc.LatinHypercube(d=n_vars, optimization="random-cd", seed=current_seed)

    sample_norm = sampler.random(n=10000)  # 每次生成1万条尝试

    # 缩放至真实物理范围
    keys = list(bounds.keys())
    lower_bounds = [bounds[k][0] for k in keys]
    upper_bounds = [bounds[k][1] for k in keys]
    sample_real = qmc.scale(sample_norm, lower_bounds, upper_bounds)
    df_batch = pd.DataFrame(sample_real, columns=keys)

    # 内部计算 r3 用于筛选
    r3_check = df_batch['r2'] - df_batch['w4']

    # 过滤: 只保留物理合法 (巴伦内径大于 0.5) 的数据
    df_valid = df_batch[r3_check >= MIN_R3_LIMIT].copy()
    valid_data = pd.concat([valid_data, df_valid], ignore_index=True)

    if batch_count % 5 == 0:
        print(f"   ...已收集 {len(valid_data)} 条有效数据")

    # 微小休眠，确保如果循环极快，下一次的时间戳种子依然会改变
    time.sleep(0.01)

# ==========================================
# 3. 截取与导出 (单文件)
# ==========================================
# 截取前 target_samples 个
df_final = valid_data.iloc[:target_samples].copy()

# 只保留 HFSS 需要的列
df_final = df_final[columns_export]

# 保留 3 位小数，避免浮点数过长
df_final = np.round(df_final, 3)

print("-" * 30)
print(f"✅ 数据准备完成！最终列: {list(df_final.columns)}")

# 添加 HFSS 识别的单位
units = {
    'L1': 'mm', 'r2': 'mm', 'w4': 'mm',
    't1': 'deg', 't2': 'deg', 'w3': 'mm'
}

df_formatted = df_final.copy()
for col in columns_export:
    df_formatted[col] = df_formatted[col].astype(str) + units[col]

# 🔥 核心修复：文件名自动增加时间戳，防止多次生成导致文件覆盖！
timestamp = time.strftime("%m%d_%H%M%S")  # 格式如：0523_143005
output_file = f"HFSS_LHS_{target_samples}_{timestamp}.txt"
print(f"\n💾 正在保存为 {output_file} ...")

df_formatted.to_csv(output_file, index=False, sep='\t')

print(f"✅ 成功！文件已保存: {output_file} (共 {len(df_formatted)} 行有效参数)")