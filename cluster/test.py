from matplotlib.font_manager import fontManager
for font in fontManager.ttflist:
    print(font.name)


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题