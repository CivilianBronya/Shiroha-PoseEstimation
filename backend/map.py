import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')  # 使用学术风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 模拟真实数据
np.random.seed(42)
epochs = np.linspace(1, 100, 100)

# 定义指数衰减拟合函数
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# 生成三组模型数据
# Model A: 快速收敛，波动小
loss_model_a = 2.5 * np.exp(-0.05 * epochs) + 0.1 + np.random.normal(0, 0.02, 100)

# Model B: 收敛较慢，波动中等
loss_model_b = 2.5 * np.exp(-0.02 * epochs) + 0.15 + np.random.normal(0, 0.05, 100)

# Model C: 不稳定，震荡大（模拟学习率过高）
loss_model_c = 2.5 * np.exp(-0.01 * epochs) + np.random.normal(0, 0.1, 100) + (epochs / 200)

# 绘图设置
plt.figure(figsize=(12, 8), dpi=100)

# 绘制三条主曲线
plt.plot(epochs, loss_model_a, label='Model A (AdamW, LR=1e-4)', color='#1f77b4', linewidth=2.5, alpha=0.8)
plt.plot(epochs, loss_model_b, label='Model B (SGD, LR=1e-2)', color='#d62728', linewidth=2.5, alpha=0.8)
plt.plot(epochs, loss_model_c, label='Model C (Unstable/High LR)', color='#2ca02c', linewidth=2, linestyle='--', alpha=0.7)

"""
到绘制预测为背代码->
"""
# 添加预测趋势（未来训练耦合点）
# 截取前80个epoch的数据进行拟合，预测后20个epoch
epochs_fit = epochs[:80]
loss_a_fit = loss_model_a[:80]

# 拟合Model A的衰减趋势
params, _ = curve_fit(exp_decay, epochs_fit, loss_a_fit, p0=[2.5, 0.05, 0.1])

# 生成未来20个epoch的预测x轴
future_epochs = np.linspace(80, 100, 20)
predicted_loss = exp_decay(future_epochs, *params)

# 绘制预测虚线
plt.plot(future_epochs, predicted_loss, color='#1f77b4', linestyle=':', linewidth=2, alpha=0.9)

# 添加理论极限（贝叶斯错误率
# 损失不可能为0，存在理论下限
theoretical_limit = 0.05
plt.axhline(y=theoretical_limit, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
plt.fill_between(epochs, theoretical_limit - 0.02, theoretical_limit + 0.02, color='gray', alpha=0.1)

# 标注与美化
plt.title('AI Training Performance Comparison: Loss Convergence', fontsize=18, pad=20, fontweight='bold')
plt.xlabel('训练次数', fontsize=14)
plt.ylabel('训练损失 (Cross-Entropy)', fontsize=14)

plt.legend(frameon=True, shadow=True, fontsize=12, loc='upper right')
plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.xlim(0, 105)
plt.ylim(0, 3)

# 添加注释
plt.annotate('Fast Convergence', xy=(20, 0.8), fontsize=12, color='#1f77b4', fontweight='bold')
plt.annotate('Slow Convergence', xy=(60, 1.5), fontsize=12, color='#d62728', fontweight='bold')
plt.annotate('Predicted Trend', xy=(85, 0.3), fontsize=10, color='#1f77b4', rotation=10)
plt.annotate('Theoretical Limit (Bayesian Error)', xy=(75, 0.08), fontsize=10, color='gray')

plt.tight_layout()
plt.show()