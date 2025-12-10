import os

# 随机种子（固定此值可保证结果复现）
RANDOM_SEED = 42  # 核心：固定随机种子

# 文件路径配置
DATA_FILE_PATH = r"D:\桌面\sci论文\数据\五种指标的性能数据集.xlsx"
RESULTS_SAVE_PATH = r"D:\桌面\sci论文\数据\预测结果\5种性能指标预测结果"

# 特征和目标列配置
FEATURE_COLUMNS = ['OMC', 'HMS', 'OCR', 'TiO2', 'AVB', 'TinosorbS', 'ZnO']
TARGET_COLUMNS = ['SPF', 'UVA-PF/SPF', 'Blue Light PF', 'UV Filter Efficiency', 'Ecological Impact']

# 绘图参数配置
PLOT_PARAMS = {
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}

# 颜色配置
COLOR_COMBINATIONS = [
    ('red', 'purple'),    # 图1: 红-紫
    ('yellow', 'green'),  # 图2: 黄-绿
    ('blue', 'orange'),   # 图3: 蓝-橙
    ('cyan', 'magenta'),  # 图4: 青-品红
    ('brown', 'gray')     # 图5: 棕-灰
]