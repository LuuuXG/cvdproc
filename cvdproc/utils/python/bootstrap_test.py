import numpy as np
from scipy.stats import ttest_ind, norm
import matplotlib.pyplot as plt

# 设置随机种子以便复现
np.random.seed(444)

# 生成两组模拟数据，符合标准正态分布
mean_1, std_1 = 0, 1
mean_2, std_2 = 0, 1
group_1 = np.random.normal(mean_1, std_1, 50)
group_2 = np.random.normal(mean_2, std_2, 50)

# 原始T检验
original_t_stat, p_value_ttest = ttest_ind(group_1, group_2, equal_var=False)

# 设置 bootstrap 次数
n_bootstrap = 5000

# 方法1：经验p值，通过置换数据进行bootstrap
t_stats_bootstrap_1 = []
for _ in range(n_bootstrap):
    # 随机置换 group_1 和 group_2 的标签
    combined = np.concatenate([group_1, group_2])
    np.random.shuffle(combined)
    new_group_1 = combined[:len(group_1)]
    new_group_2 = combined[len(group_1):]

    # 计算置换后的 t 值
    t_stat, _ = ttest_ind(new_group_1, new_group_2, equal_var=False)
    t_stats_bootstrap_1.append(t_stat)

# 计算经验p值（双边检验）
p_value_method_1 = np.mean(np.abs(t_stats_bootstrap_1) >= np.abs(original_t_stat))

# 方法2：基于标准差重新计算的Z值
t_stats_bootstrap_2 = []
for _ in range(n_bootstrap):
    # 随机从 group_1 和 group_2 中有放回抽样
    new_group_1 = np.random.choice(group_1, size=len(group_1), replace=True)
    new_group_2 = np.random.choice(group_2, size=len(group_2), replace=True)

    # 计算新的 t 值
    t_stat, _ = ttest_ind(new_group_1, new_group_2, equal_var=False)
    t_stats_bootstrap_2.append(t_stat)

# 计算基于标准差的理论p值
std_bootstrap_2 = np.std(t_stats_bootstrap_2)
new_z_stat = original_t_stat / std_bootstrap_2
p_value_method_2 = 2 * (1 - norm.cdf(np.abs(new_z_stat)))

# 输出结果
print(f"Original t-statistic: {original_t_stat}")
print(f"Method 1 (empirical p-value): {p_value_method_1}")
print(f"Method 2 (theoretical p-value): {p_value_method_2}")
print(f"Traditional t-test p-value: {p_value_ttest}")

# 可视化两种方法的t值分布
plt.hist(t_stats_bootstrap_1, bins=30, alpha=0.5, label='Bootstrap T-values (Method 1)')
plt.hist(t_stats_bootstrap_2, bins=30, alpha=0.5, label='Bootstrap T-values (Method 2)')
plt.axvline(original_t_stat, color='red', linestyle='dashed', linewidth=2, label='Original T-value')
plt.legend()
plt.xlabel('T-value')
plt.ylabel('Frequency')
plt.title('Distribution of T-values from Bootstrap')
plt.show()
