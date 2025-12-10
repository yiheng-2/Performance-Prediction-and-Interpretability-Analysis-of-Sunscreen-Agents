import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """数据处理类，负责数据加载、清洗和预处理"""
    
    def __init__(self, feature_columns, target_columns):
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.scalers = {}  # 存储每个目标变量的标准化器
    
    def load_data(self, file_path):
        """加载数据并进行初步清洗"""
        try:
            df = pd.read_excel(file_path)
            print("Data loaded successfully!")
            print(f"Data shape: {df.shape}")
            
            # 检查并删除空列
            df = df.dropna(axis=1, how='all')
            print(f"Cleaned data shape: {df.shape}")
            
            # 确保列存在
            self.feature_columns = [col for col in self.feature_columns if col in df.columns]
            self.target_columns = [col for col in self.target_columns if col in df.columns]
            
            X = df[self.feature_columns]
            y = df[self.target_columns]
            
            print(f"Features: {X.shape[1]}")
            print(f"Targets: {y.shape[1]}")
            print(f"Target names: {self.target_columns}")
            
            return X, y
            
        except Exception as e:
            print(f"Data loading failed: {e}")
            return None, None
    
    def split_and_scale(self, X, y):
        """分割数据并进行标准化"""
        X_train_dict = {}
        X_test_dict = {}
        y_train_dict = {}
        y_test_dict = {}
        
        for i, target in enumerate(self.target_columns):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y.iloc[:, i], test_size=0.2, random_state=42
            )
            
            # 标准化特征
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            # 标准化目标变量
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
            
            X_train_dict[target] = (X_train_scaled, X_train)
            X_test_dict[target] = (X_test_scaled, X_test)
            y_train_dict[target] = (y_train_scaled, y_train)
            y_test_dict[target] = (y_test_scaled, y_test)
            self.scalers[target] = (scaler_X, scaler_y)
        
        return X_train_dict, X_test_dict, y_train_dict, y_test_dict
    
    @staticmethod
    def clean_filename(filename):
        """清理文件名，移除不允许的字符"""
        cleaned = re.sub(r'[\\/*?:"<>|()]', "_", filename)
        cleaned = re.sub(r'\s+', '_', cleaned)
        cleaned = re.sub(r'_+', '_', cleaned)
        return cleaned.strip('_')