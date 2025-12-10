import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 直接在可视化模块中定义颜色配置
# 5种不同的颜色组合（鲜艳且区分度高）
COLOR_COMBINATIONS = [
    ('#FF3333', '#CC00CC'),    # 亮红-亮紫
    ('#FFFF33', '#33CC33'),    # 亮黄-亮绿
    ('#3333FF', '#FF9933'),    # 亮蓝-亮橙
    ('#33FFFF', '#FF33FF'),    # 亮青-亮品红
    ('#CC6600', '#666666')     # 亮棕-深灰
]

# 模型对比的颜色
MODEL_COLORS = {
    'RandomForest': '#FF5733',  # 亮橙红
    'XGBoost': '#33FF57',       # 亮绿
    'DNN': '#3357FF'            # 亮蓝
}

# 热图配色方案
HEATMAP_CMAPS = {
    'r2': 'YlOrRd',            # 红橙渐变
    'mae': 'YlGnBu_r',         # 蓝绿渐变（反色）
    'rmse': 'PuBuGn_r',        # 紫蓝绿渐变（反色）
    'correlation': 'RdBu_r',   # 红蓝对比（反色）
    'shap': 'YlOrRd'           # SHAP图配色
}

# 设置绘图参数（SCI期刊风格 + 增强颜色显示）
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.facecolor': 'white',  # 白色背景增强颜色对比度
    'figure.facecolor': 'white'
})

class Visualizer:
    """可视化工具类（颜色配置集成在内部）"""
    
    @staticmethod
    def plot_training_history(history, target):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'DNN Training Process - {target}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE曲线
        ax2.plot(history['mae'], 'b-', label='Training MAE', linewidth=2)
        ax2.plot(history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE')
        ax2.set_title(f'DNN MAE Progression - {target}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(y_train_actual, y_train_pred, y_test, y_pred, target, model_name, metrics, target_names):
        """绘制预测结果散点图（使用内部定义的颜色组合）"""
        try:
            # 根据目标在列表中的索引获取对应的颜色组合
            target_idx = target_names.index(target)
            train_color, test_color = COLOR_COMBINATIONS[target_idx % len(COLOR_COMBINATIONS)]
        except (ValueError, IndexError):
            # 异常情况使用默认颜色
            train_color, test_color = COLOR_COMBINATIONS[0]
        
        plt.figure(figsize=(8, 6))
        
        # 训练集散点
        plt.scatter(
            y_train_actual, y_train_pred, 
            alpha=0.7,
            color=train_color, 
            label='Training set', 
            s=60,
            edgecolors='black', 
            linewidth=0.8
        )
        
        # 测试集散点
        plt.scatter(
            y_test, y_pred, 
            alpha=0.7, 
            color=test_color, 
            label='Test set', 
            s=60, 
            edgecolors='black', 
            linewidth=0.8
        )
        
        # 理想线
        min_val = min(min(y_train_actual), min(y_test))
        max_val = max(max(y_train_actual), max(y_test))
        plt.plot(
            [min_val, max_val], [min_val, max_val], 
            'k--', 
            alpha=0.9,
            linewidth=2.5,
            label='Ideal fit'
        )
        
        plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.title(f'{target} Prediction - {model_name} Model', fontsize=14, fontweight='bold')
        
        # 指标文本
        textstr = f'R² = {metrics["r2"]:.3f}\nMAE = {metrics["mae"]:.3f}\nRMSE = {metrics["rmse"]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_data):
        """绘制模型指标对比图"""
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R²对比
        r2_pivot = metrics_df.pivot(index='Target', columns='Model', values='R²')
        sns.heatmap(
            r2_pivot, 
            annot=True, 
            fmt='.3f', 
            cmap=HEATMAP_CMAPS['r2'], 
            ax=axes[0], 
            cbar_kws={'label': 'R²'},
            vmin=0, vmax=1
        )
        axes[0].set_title('R² Comparison', fontweight='bold')
        
        # MAE对比
        mae_pivot = metrics_df.pivot(index='Target', columns='Model', values='MAE')
        sns.heatmap(
            mae_pivot, 
            annot=True, 
            fmt='.3f', 
            cmap=HEATMAP_CMAPS['mae'], 
            ax=axes[1], 
            cbar_kws={'label': 'MAE'}
        )
        axes[1].set_title('MAE Comparison', fontweight='bold')
        
        # RMSE对比
        rmse_pivot = metrics_df.pivot(index='Target', columns='Model', values='RMSE')
        sns.heatmap(
            rmse_pivot, 
            annot=True, 
            fmt='.3f', 
            cmap=HEATMAP_CMAPS['rmse'], 
            ax=axes[2], 
            cbar_kws={'label': 'RMSE'}
        )
        axes[2].set_title('RMSE Comparison', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(pred_data, actual_data):
        """绘制相关性热图"""
        pred_df = pd.DataFrame(pred_data)
        actual_df = pd.DataFrame(actual_data)
        corr_matrix = pd.concat([pred_df, actual_df], axis=1).corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt='.2f', 
            cmap=HEATMAP_CMAPS['correlation'], 
            center=0, 
            square=True, 
            cbar_kws={'label': 'Pearson Correlation'},
            vmin=-1, vmax=1
        )
        plt.title('Pearson Correlation Heatmap of Predictions and Actual Values', fontweight='bold')
        plt.tight_layout()
        plt.show()