import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    """Prepara os dados para serem usados no modelo de machine learning."""
    def __init__(self, df):
        self.df = df
        self.encoder_columns = None
        self.label_encoders = {}

    def preprocess(self, feature_cols, target_col):
        """Executa a engenharia de features e o one-hot encoding."""
        # Cria a feature 'mes' a partir da data
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['mes'] = self.df['data'].dt.month
        
        # Adicionar mais features temporais para melhorar a precisão
        self.df['dia'] = self.df['data'].dt.day
        self.df['trimestre'] = self.df['data'].dt.quarter
        
        # Garantir que temos identificação de produtos, se existir
        if 'id_produto' in self.df.columns and 'id_produto' not in feature_cols:
            feature_cols.append('id_produto')
            
        if 'nome_produto' in self.df.columns and 'nome_produto' not in feature_cols:
            # Aplicar Label Encoding em nome_produto
            le = LabelEncoder()
            self.df['nome_produto_encoded'] = le.fit_transform(self.df['nome_produto'])
            self.label_encoders['nome_produto'] = le
            feature_cols.append('nome_produto_encoded')
        
        X = self.df[feature_cols]
        y = self.df[target_col]

        # Converte colunas categóricas em numéricas
        categorical_cols = ['categoria', 'promocao', 'dia_da_semana']
        
        # Adicionar produto_id na lista de categóricas se existir
        if 'id_produto' in X.columns:
            categorical_cols.append('id_produto')
            
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        self.encoder_columns = X_encoded.columns # Salva a ordem das colunas
        
        print("Pré-processamento e engenharia de features concluídos.")
        print(f"Features finais: {len(X_encoded.columns)} colunas")
        return X_encoded, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Divide os dados em conjuntos de treino e teste."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def encode_new_data(self, new_data):
        """Codifica novos dados usando os mesmos encoders do treinamento."""
        # Aplicar label encoding para colunas que já foram codificadas
        for col, encoder in self.label_encoders.items():
            if col in new_data.columns:
                new_data[f'{col}_encoded'] = encoder.transform(new_data[col])
        
        return new_data