from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .base_model import BaseModel

class DNNModel(BaseModel):
    """深度神经网络模型"""
    
    def __init__(self, input_dim, output_dim=1):
        super().__init__("DNN")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history = None
        self._build_model()
        
    def _build_model(self):
        """构建模型结构"""
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(12, activation='relu'),
            Dense(self.output_dim)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
    def train(self, X_train, y_train, validation_split=0.2, epochs=200, batch_size=16):
        """训练模型"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[early_stopping]
        )
        self.results['history'] = self.history
        return self.history
        
    def predict(self, X):
        """预测"""
        return self.model.predict(X).flatten()