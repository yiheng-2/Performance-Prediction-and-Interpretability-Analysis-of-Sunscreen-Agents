from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """随机森林模型"""
    
    def __init__(self):
        super().__init__("RandomForest")
        self._build_model()
        
    def _build_model(self):
        """构建模型"""
        self.model = RandomForestRegressor(
            n_estimators=20, 
            random_state=42, 
            max_depth=4
        )
        
    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """预测"""
        return self.model.predict(X)