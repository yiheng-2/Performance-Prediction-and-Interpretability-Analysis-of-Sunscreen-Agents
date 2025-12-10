import os
import numpy as np
import pandas as pd
from src.data.data_processor import DataProcessor
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.dnn_model import DNNModel
from src.visualization.visualizer import Visualizer
from src.analysis.shap_analyzer import SHAPAnalyzer
from config import DATA_FILE_PATH, RESULTS_SAVE_PATH, FEATURE_COLUMNS, TARGET_COLUMNS

class SunscreenPredictor:
    """防晒霜性能预测器主类"""
    
    def __init__(self):
        self.data_processor = DataProcessor(FEATURE_COLUMNS, TARGET_COLUMNS)
        self.models = {}  # 存储每个目标的模型
        self.results = {}  # 存储每个目标的结果
        self.best_models = {}  # 存储每个目标的最佳模型
        self.X = None
        self.y = None
        self.X_train_dict = None
        self.X_test_dict = None
        self.y_train_dict = None
        self.y_test_dict = None
        
    def load_and_prepare_data(self):
        """加载并准备数据"""
        self.X, self.y = self.data_processor.load_data(DATA_FILE_PATH)
        
        if self.X is not None:
            print("\nPreparing data...")
            self.X_train_dict, self.X_test_dict, self.y_train_dict, self.y_test_dict = \
                self.data_processor.split_and_scale(self.X, self.y)
            return True
        return False
    
    def train_models(self):
        """训练所有模型"""
        print("\nTraining models...")
        
        for target in self.data_processor.target_columns:
            print(f"\nTraining models for: {target}")
            self.models[target] = {}
            self.results[target] = {}
            
            X_train_scaled, X_train_orig = self.X_train_dict[target]
            X_test_scaled, X_test_orig = self.X_test_dict[target]
            y_train_scaled, y_train_orig = self.y_train_dict[target]
            y_test_scaled, y_test_orig = self.y_test_dict[target]
            scaler_X, scaler_y = self.data_processor.scalers[target]
            
            best_r2 = -np.inf
            best_model_name = None
            
            # 初始化模型
            model_instances = {
                'RandomForest': RandomForestModel(),
                'XGBoost': XGBoostModel(),
                'DNN': DNNModel(input_dim=X_train_scaled.shape[1])
            }
            
            for model_name, model in model_instances.items():
                print(f"  Training {model_name}...")
                
                try:
                    # 训练模型
                    if model_name == 'DNN':
                        model.train(X_train_scaled, y_train_scaled)
                        self.results[target][model_name] = {'history': model.history.history}
                    else:
                        model.train(X_train_scaled, y_train_scaled)
                        self.results[target][model_name] = {}
                    
                    # 预测
                    y_pred_scaled = model.predict(X_test_scaled)
                    
                    # 保存模型
                    self.models[target][model_name] = model
                    
                    # 反标准化预测结果
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    y_test_original = y_test_orig.values
                    
                    # 评估
                    metrics = model.evaluate(y_test_original, y_pred)
                    self.results[target][model_name].update(metrics)
                    
                    print(f"    {model_name} - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
                    
                    # 更新最佳模型
                    if metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_model_name = model_name
                        self.best_models[target] = {
                            'model': model,
                            'model_name': model_name,
                            'r2': metrics['r2'],
                            'mae': metrics['mae'],
                            'rmse': metrics['rmse'],
                            'y_test': metrics['y_true'],
                            'y_pred': metrics['y_pred'],
                            'X_train': X_train_scaled,
                            'X_test': X_test_scaled,
                            'X_test_original': X_test_orig,
                            'scaler_X': scaler_X,
                            'scaler_y': scaler_y
                        }
                    
                except Exception as e:
                    print(f"    {model_name} training failed: {e}")
                    continue
    
    def visualize_results(self):
        """可视化结果（调整为：先R²热力图 → 散点图 → 再复制一张R²热力图）"""
        print("\nGenerating visualizations...")
        
        # 1. 先生成所有指标数据（供后续复用）
        metrics_data = []
        for target in self.data_processor.target_columns:
            for model_name in ['RandomForest', 'XGBoost', 'DNN']:
                if model_name in self.results.get(target, {}):
                    metrics = self.results[target][model_name]
                    metrics_data.append({
                        'Target': target,
                        'Model': model_name,
                        'R²': metrics['r2'],
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse']
                    })
        
        # 2. 绘制第一张R²热力图（仅R²，无其他指标）
        self._plot_single_metric_heatmap(metrics_data, metric='R²', title='R² Comparison')
        
        # 3. 可视化训练过程（DNN的损失和MAE曲线）
        for target in self.data_processor.target_columns:
            if 'DNN' in self.results.get(target, {}):
                Visualizer.plot_training_history(self.results[target]['DNN']['history'], target)
                break  # 只显示第一个目标的训练过程
        
        # 4. 绘制预测结果散点图
        for target in self.data_processor.target_columns:
            if target in self.best_models:
                best_model = self.best_models[target]
                model_name = best_model['model_name']
                scaler_y = best_model['scaler_y']
                
                # 获取训练集预测结果
                if model_name == 'DNN':
                    y_train_pred_scaled = best_model['model'].predict(best_model['X_train']).flatten()
                else:
                    y_train_pred_scaled = best_model['model'].predict(best_model['X_train'])
                
                y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
                y_train_actual = scaler_y.inverse_transform(self.y_train_dict[target][0].reshape(-1, 1)).flatten()
                
                # 绘制预测图
                metrics = {
                    'r2': best_model['r2'],
                    'mae': best_model['mae'],
                    'rmse': best_model['rmse']
                }
                
                Visualizer.plot_predictions(
                    y_train_actual, y_train_pred,
                    best_model['y_test'], best_model['y_pred'],
                    target, model_name, 
                    metrics,
                    self.data_processor.target_columns
                )
        
        # 5. 在散点图后复制一张R²热力图（与第一张相同）
        self._plot_single_metric_heatmap(metrics_data, metric='R²', title='R² Comparison (Duplicate)')
       
        # 6. 相关性热图（保持在最后）
        pred_data = {}
        actual_data = {}
        for target in self.data_processor.target_columns:
            if target in self.best_models:
                pred_data[f'{target}_pred'] = self.best_models[target]['y_pred']
                actual_data[f'{target}_actual'] = self.best_models[target]['y_test']
        Visualizer.plot_correlation_heatmap(pred_data, actual_data)
        
    

    # 新增：单独绘制单个指标（如R²）热力图的方法
    def _plot_single_metric_heatmap(self, metrics_data, metric='R²', title='R² Comparison'):
        """绘制单个指标的热力图（仅R²）"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from src.visualization.visualizer import HEATMAP_CMAPS  # 复用颜色配置
        
        metrics_df = pd.DataFrame(metrics_data)
        metric_pivot = metrics_df.pivot(index='Target', columns='Model', values=metric)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metric_pivot,
            annot=True,
            fmt='.3f',
            cmap=HEATMAP_CMAPS.get(metric.lower(), 'YlOrRd'),  # 用R²的颜色配置
            cbar_kws={'label': metric},
            vmin=0, vmax=1 if metric == 'R²' else None  # R²范围固定0-1
        )
        plt.title(title, fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def perform_shap_analysis(self):
        """执行SHAP分析"""
        print("\nPerforming SHAP analysis...")
        shap_data = []
        
        for target in self.data_processor.target_columns:
            if target in self.best_models:
                best_model = self.best_models[target]
                model = best_model['model']
                model_name = best_model['model_name']
                X_test_scaled = best_model['X_test']
                X_test_original = best_model['X_test_original']
                
                print(f"Performing SHAP analysis for {target} using {model_name}")
                
                # 关键修改：传递底层模型而不是包装器
                if model_name in ['RandomForest', 'XGBoost']:
                    # 对于树模型，传递底层的sklearn/xgboost模型
                    underlying_model = model.model
                else:
                    # 对于DNN，传递整个模型（需要predict方法）
                    underlying_model = model
                
                # 执行SHAP分析
                shap_values = SHAPAnalyzer.analyze_model(
                    underlying_model, model_name, X_test_scaled, 
                    X_test_original, self.data_processor.feature_columns, target
                )
                
                # 收集SHAP数据用于综合报告
                if shap_values is not None:
                    try:
                        # 处理不同的SHAP值格式
                        if hasattr(shap_values, 'shape'):
                            if len(shap_values.shape) == 2:
                                mean_abs_shap = np.abs(shap_values).mean(0)
                            else:
                                mean_abs_shap = np.abs(shap_values)
                        else:
                            mean_abs_shap = np.abs(np.array(shap_values))
                        
                        # 确保mean_abs_shap是正确形状
                        if hasattr(mean_abs_shap, 'shape') and len(mean_abs_shap.shape) > 0:
                            if len(mean_abs_shap) == len(self.data_processor.feature_columns):
                                # 正常情况
                                for i, feature in enumerate(self.data_processor.feature_columns):
                                    shap_data.append({
                                        'Target': target,
                                        'Feature': feature,
                                        'Mean|SHAP|': float(mean_abs_shap[i]),
                                        'Model': model_name
                                    })
                            else:
                                # 形状不匹配，使用平均值
                                avg_shap = np.mean(mean_abs_shap)
                                for feature in self.data_processor.feature_columns:
                                    shap_data.append({
                                        'Target': target,
                                        'Feature': feature,
                                        'Mean|SHAP|': float(avg_shap),
                                        'Model': model_name
                                    })
                        else:
                            # 标量情况
                            for feature in self.data_processor.feature_columns:
                                shap_data.append({
                                    'Target': target,
                                    'Feature': feature,
                                    'Mean|SHAP|': float(mean_abs_shap),
                                    'Model': model_name
                                })
                                
                    except Exception as e:
                        print(f"Error processing SHAP values for {target}: {e}")
                        # 添加默认值避免报告生成失败
                        for feature in self.data_processor.feature_columns:
                            shap_data.append({
                                'Target': target,
                                'Feature': feature,
                                'Mean|SHAP|': 0.0,
                                'Model': model_name
                            })
        
        # 创建综合SHAP报告
        if shap_data:
            SHAPAnalyzer.create_comprehensive_report(
                shap_data, self.data_processor.feature_columns, 
                self.data_processor.target_columns
            )
        else:
            print("No SHAP data collected for comprehensive report")
    def save_results(self):
        """保存预测结果"""
        os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
        
        # 保存每个目标变量的预测结果
        for target in self.data_processor.target_columns:
            results_data = {}
            
            # 添加实际值
            if target in self.best_models:
                results_data['Actual'] = self.best_models[target]['y_test']
                results_data[f'Predicted_{self.best_models[target]["model_name"]}'] = self.best_models[target]['y_pred']
            
            # 添加其他模型的预测结果
            for model_name in ['RandomForest', 'XGBoost', 'DNN']:
                if model_name in self.results.get(target, {}) and model_name != self.best_models.get(target, {}).get('model_name'):
                    results_data[f'Predicted_{model_name}'] = self.results[target][model_name]['y_pred']
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                safe_filename = self.data_processor.clean_filename(f'{target}_predictions')
                results_df.to_excel(f'{RESULTS_SAVE_PATH}/{safe_filename}.xlsx', index=False)
                print(f"Saved: {safe_filename}.xlsx")
        
        # 保存模型评估指标
        metrics_data = []
        for target in self.data_processor.target_columns:
            for model_name in ['RandomForest', 'XGBoost', 'DNN']:
                if model_name in self.results.get(target, {}):
                    metrics = self.results[target][model_name]
                    metrics_data.append({
                        'Target': target,
                        'Model': model_name,
                        'R²': metrics['r2'],
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse']
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(f'{RESULTS_SAVE_PATH}/model_metrics.xlsx', index=False)
        print("Saved: model_metrics.xlsx")
        
        print(f"\nAll results saved to: {RESULTS_SAVE_PATH}")

def main():
    """主函数"""
    predictor = SunscreenPredictor()
    
    # 加载和准备数据
    if predictor.load_and_prepare_data():
        # 训练模型
        predictor.train_models()
        
        # 可视化结果
        predictor.visualize_results()
        
        # 执行SHAP分析
        predictor.perform_shap_analysis()
        
        # 保存结果
        predictor.save_results()
        
        print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()