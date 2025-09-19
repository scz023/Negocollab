import matplotlib.pyplot as plt


TICK_SIZE = 22
LABEL_SIZE = 27
LEGEND_SIZE = 22

# 数据
categories = ['m1','m2','m3','m4']
nego = [2.266, 0.764, 0.711, 0.722]
baseline = [24.897, 13.704, 13.28, 12.32]

# 设置柱状图宽度
bar_width = 0.39

# 设置x轴位置
r1 = [i for i in range(len(categories))]
r2 = [i + bar_width for i in r1]

# 绘制柱状图
plt.bar(r1, nego, width=bar_width, label='NegoCollab', color='forestgreen')
plt.bar(r2, baseline, width=bar_width, label='Baseline', color='orange')

# 设置x轴刻度标签
plt.xticks([r + bar_width / 2 for r in r1], categories)


# 添加标题和坐标轴标签
# plt.title('图表标题')
plt.xlabel('Agent Type', fontsize=LABEL_SIZE)
plt.ylabel('KL Divergence', fontsize=LABEL_SIZE)

# 增大 x、y 轴刻度标签的字体大小并去掉刻度线
plt.tick_params(axis='both', which='both', length=0, labelsize=13.2)

# 增大 x、y 轴刻度标签的字体大小
plt.xticks(fontsize=TICK_SIZE)


original_yticks = plt.yticks()[0]
# 每隔一个刻度显示一个（取一半）
new_yticks = original_yticks[:-1:2]
plt.yticks(new_yticks, fontsize=TICK_SIZE)
# plt.yticks(fontsize=TICK_SIZE)

# 添加图例
plt.legend(fontsize=LEGEND_SIZE)

# 加粗表外框线
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black') 
    spine.set_linewidth(2)

# 去掉图周围的空白
plt.tight_layout()

# 显示图表
plt.show()

plt.savefig('domain_gap.png')
