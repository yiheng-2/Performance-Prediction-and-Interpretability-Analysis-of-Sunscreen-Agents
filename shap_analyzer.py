import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class SHAPAnalyzer:
    """SHAP分析工具类"""
    
    @staticmethod
    def analyze_model(model, model_name, X_test_scaled, X_test_original, feature_names, target):
        """分析单个模型的SHAP值"""
        try:
            print(f"\nSHAP analysis for {target} (Best model: {model_name})")
            print(f"Model type received: {type(model)}")
            
            if model_name in ['RandomForest', 'XGBoost']:
                # 现在model已经是底层的sklearn/xgboost模型
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
                
                # 处理可能的多种输出格式
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # 回归问题取第一个
                elif hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
                    shap_values = shap_values[:, :, 0]  # 取第一个输出维度
                
                print(f"SHAP values shape: {np.array(shap_values).shape}")
                
                # 创建SHAP摘要图
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_original, 
                                feature_names=feature_names, 
                                show=False, plot_size=None)
                plt.title(f'SHAP Summary Plot for {target}\n({model_name} Model)', 
                        fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                return shap_values
                
            elif model_name == 'DNN':
                # DNN模型使用KernelExplainer
                sample_size = min(50, X_test_scaled.shape[0])
                background = X_test_scaled[np.random.choice(X_test_scaled.shape[0], 
                                                        sample_size, 
                                                        replace=False)]
                explainer = shap.KernelExplainer(model.predict, background)
                
                # 限制样本数量以提高性能
                sample_size_pred = min(100, X_test_scaled.shape[0])
                shap_values = explainer.shap_values(X_test_scaled[:sample_size_pred])
                
                # 创建SHAP摘要图
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_original[:sample_size_pred], 
                                feature_names=feature_names, 
                                show=False, plot_size=None)
                plt.title(f'SHAP Summary Plot for {target}\n({model_name} Model)', 
                        fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                return shap_values
                
        except Exception as e:
            print(f"SHAP analysis failed for {target} with {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def create_comprehensive_report(shap_data, feature_names, target_names):
        """创建综合的SHAP分析报告"""
        if not shap_data:
            print("No SHAP data available for comprehensive report")
            return
            
        print("\nCreating comprehensive SHAP analysis report...")
        
        shap_df = pd.DataFrame(shap_data)
        
        # 创建热图显示特征对目标的影响
        pivot_df = shap_df.pivot(index='Feature', columns='Target', values='Mean|SHAP|')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Mean |SHAP value| (Feature Importance)'})
        plt.title('Comprehensive SHAP Analysis: Sunscreen Ingredients Impact on Performance Metrics', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Performance Metrics')
        plt.ylabel('Sunscreen Ingredients')
        plt.tight_layout()
        plt.show()
        
        # 创建特征重要性对比图
        plt.figure(figsize=(12, 6))
        for target in target_names:
            target_data = shap_df[shap_df['Target'] == target]
            if not target_data.empty:
                plt.bar(target_data['Feature'], target_data['Mean|SHAP|'], 
                       alpha=0.7, label=target)
        
        plt.xlabel('Sunscreen Ingredients')
        plt.ylabel('Mean |SHAP value|')
        plt.title('Comparative Feature Importance Across Different Performance Metrics', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()