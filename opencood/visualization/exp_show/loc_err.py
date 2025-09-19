import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
import pandas as pd
import matplotlib.pyplot as plt

"""
MPDA PnPDA MPDA-P PnPDA-P STAMP NegoCollab NegoCollab-FT
定义颜色列表，你可以根据需要修改
"""
colors = ['goldenrod', 'royalblue', 'gold', 'cornflowerblue', 'brown', 'forestgreen', 'green', ]
# colors = ['gold', 'cornflowerblue', 'goldenrod', 'royalblue', 'brown', 'darkgreen', 'green']


LABEL_SIZE = 45
TICK_SIZE = 33
LEGEND_SIZE = 25
LINE_WIDTH = 8
MARKSIZE=12
LEGEND_WIDTH = 6 
NCOL = 2

filename = 'locerr_m1m2_ap50'
ap = 'AP@0.5'

# filename = 'locerr_m1m2_ap70'
# ap = 'AP@0.7'


# filename = 'locerr_all_ap50'
# filename = 'locerr_all_ap70'

# 读取 CSV 文件
try:
    df = pd.read_csv(f'{filename}.csv', index_col=0)
except FileNotFoundError:
    print("错误：未找到 CSV 文件，请检查文件路径和文件名。")
else:
    # 获取定位误差（第一行数据）
    pose_noise_std = df.columns.astype(float)

    # 创建画布
    plt.figure(figsize=(10, 8))

    # 遍历每一种方法
    for i, (method, values) in enumerate(df.iterrows()):
        # 绘制折线图，设置颜色和数据点样式
        plt.plot(pose_noise_std, values, linewidth=LINE_WIDTH, \
            markersize=MARKSIZE, marker='o', \
            color=colors[i], label=method)

    # 设置图表标题和坐标轴标签
    # plt.title('Method Performance vs Pose Noise Std')
    plt.xlabel('Pose Noise Std', fontsize=LABEL_SIZE)
    plt.ylabel(ap, fontsize=LABEL_SIZE)

    # 显示图例
    plt.legend(ncol=NCOL, fontsize=LEGEND_SIZE)
    
    # 固定 y 轴范围，这里假设下限为 0，上限为 1，你可以根据实际情况修改
    plt.ylim(0.15, 0.9)
    # plt.xlim(0, 0.9)
    
    
    # 获取当前x轴刻度位置
    original_xticks = plt.xticks()[0]
    # 每隔一个刻度显示一个（取一半）
    new_xticks = original_xticks[1:-1:2]
    # 增大 x、y 轴刻度标签的字体大小
    plt.xticks(new_xticks, fontsize=TICK_SIZE)
    
    original_yticks = plt.yticks()[0]
    # 每隔一个刻度显示一个（取一半）
    new_yticks = original_yticks[::2]
    plt.yticks(new_yticks, fontsize=TICK_SIZE)
    
    # plt.yticks(fontsize=TICK_SIZE)
    
   # 加粗表外框线
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 去掉图周围的空白
    plt.tight_layout()

    # 显示网格线
    plt.grid(True)

    # 显示图表
    plt.show()
    
    plt.savefig(f'{filename}.png')
    