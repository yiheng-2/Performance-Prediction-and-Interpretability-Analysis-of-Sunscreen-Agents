import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class BaseModel:
    """所有模型的基类"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.results = {}
        
    def train(self, X_train, y_train):
        """训练模型，需在子类中实现"""
        raise NotImplementedError("Subclasses must implement train method")
        
    def predict(self, X):
        """预测，需在子类中实现"""
        raise NotImplementedError("Subclasses must implement predict method")
        
    def evaluate(self, y_true, y_pred):
        """评估模型性能"""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'y_true': y_true,
            'y_pred': y_pred
        }