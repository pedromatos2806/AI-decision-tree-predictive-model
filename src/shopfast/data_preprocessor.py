import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Prepara os dados para serem usados no modelo de machine learning."""
    def __init__(self, df):
        self.df = df
        self.encoder_columns = None

    def preprocess(self, feature_cols, target_col):
        """Executa a engenharia de features e o one-hot encoding."""
        # Cria a feature 'mes' a partir da data
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['mes'] = self.df['data'].dt.month
        
        X = self.df[feature_cols]
        y = self.df[target_col]

        # Converte colunas categóricas em numéricas
        X_encoded = pd.get_dummies(X, columns=['categoria', 'promocao', 'dia_da_semana'], drop_first=True)
        self.encoder_columns = X_encoded.columns # Salva a ordem das colunas
        
        print("Pré-processamento e engenharia de features concluídos.")
        return X_encoded, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Divide os dados em conjuntos de treino e teste."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)