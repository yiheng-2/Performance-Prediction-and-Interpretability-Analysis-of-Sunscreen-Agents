import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost模型"""
    
    def __init__(self):
        super().__init__("XGBoost")
        self._build_model()
        
    def _build_model(self):
        """构建模型"""
        self.model = xgb.XGBRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=6
        )
        
    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """预测"""
        return self.model.predict(X)