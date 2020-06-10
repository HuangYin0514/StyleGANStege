import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 柱高信息
Y = [0.11, 0.0670]
Y1 = [0.02, 0.018]
X = np.arange(len(Y))



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


# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(2)  # 柱状图在横坐标上的位置
# # 列出你要显示的数据，数据的列表长度与x长度相同
# y1 = [0.11, 0.12]
# y2 = [0.02, 0.03]


# bar_width = 0.3  # 设置柱状图的宽度
# tick_label = ['out', 'DCGAN']

# # 绘制并列柱状图
# plt.bar(x, y1, bar_width, color='w', label='AAAAAAA', edgecolor="k")
# plt.bar(x+bar_width, y2, bar_width, color='w', label='BBBBBBB', edgecolor="k", hatch="\\\\\\\\\\")

# plt.yticks(np.arange(0, 0.2, 0.02))
# plt.xticks(x+bar_width/2, tick_label)  # 显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# plt.legend(loc='best')
# plt.show()

savepath = 'experiments/BerCuver/hist.jpg'
plt.savefig(savepath)
