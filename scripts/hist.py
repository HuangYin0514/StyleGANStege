import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



plt.figure(figsize=(5,4))
# 柱高信息
Y = [0.11, 0.0670]
Y1 = [0.016, 0.018]
X = np.arange(2)



bar_width = 0.25
tick_label = ['our', 'DCGAN']

# 显示每个柱的具体高度
for x, y in zip(X, Y):
    plt.text(x+0.005, y, '%.3f' % y, ha='center', va='bottom')

for x, y1 in zip(X, Y1):
    plt.text(x+0.24, y1, '%.3f' % y1, ha='center', va='bottom')

# 绘制柱状图
plt.bar(X, Y, bar_width, align="center", color="w", edgecolor="k", label="AUC")
plt.bar(X+bar_width, Y1, bar_width, align="center", edgecolor="k", color="w", hatch="\\\\\\\\\\", label="MAP")

plt.xticks(X+bar_width/2, tick_label)
plt.yticks(np.arange(0, 0.2, 0.02))
# 显示图例
plt.legend()


savepath = 'experiments/BerCuver/hist.jpg'
plt.savefig(savepath)
