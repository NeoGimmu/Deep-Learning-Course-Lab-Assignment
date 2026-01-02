import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class BostonHousingLoader:
    """Boston Housing 数据集加载器（使用替代方案）"""
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler() # 特征标准化器
    
    def load_data(self):
        """
        加载数据（替代方案：从UCI 或openml 获取）
        :return: X_train, y_train, X_test, y_test（均已标准化）
        """
        try:
            # 方案1：从openml 加载（需sklearn≥0.24）
            from sklearn.datasets import fetch_openml
            data = fetch_openml(name="boston", version=1,
            as_frame=True, parser="pandas")
            X = data.data.values
            y = data.target.values.astype(np.float32)
        except:
        # 方案2：从UCI 下载（备用）
            url = "https://archive.ics.uci.edu/ml/machinelearning-databases/housing/housing.data"
            columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT','MEDV']
            data = pd.read_csv(url, sep='\s+', names=columns)
            X = data.drop('MEDV', axis=1).values
            y = data['MEDV'].values.astype(np.float32)
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,random_state=self.random_state, shuffle=True)
        # 特征标准化（仅对训练集拟合，避免数据泄露）
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, y_train, X_test, y_test
    
    def preprocess_new_data(self, X):
        """预处理新数据（使用训练集的标准化参数）"""
        return self.scaler.transform(X.reshape(-1, X.shape[-1]))