import pandas as pd
import numpy as np

class Predictor:
    """Usa um modelo treinado para fazer novas previsões."""
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns

    def predict_demand(self, new_sale_data):
        """Prevê a demanda para um novo conjunto de dados de venda."""
        # Cria um DataFrame a partir dos novos dados
        new_sale_df = pd.DataFrame(new_sale_data, index=[0])
        
        # Processar a coluna 'mes' se ela não existir, mas 'data' existir
        if 'mes' not in new_sale_df.columns and 'data' in new_sale_df.columns:
            try:
                new_sale_df['mes'] = pd.to_datetime(new_sale_df['data']).dt.month
            except:
                # Se falhar, use um valor padrão
                new_sale_df['mes'] = 1
        
        # Aplica o one-hot encoding nas colunas categóricas
        if 'categoria' in new_sale_df.columns:
            new_sale_df = pd.get_dummies(new_sale_df, columns=['categoria'], prefix='categoria')
        
        if 'promocao' in new_sale_df.columns:
            new_sale_df = pd.get_dummies(new_sale_df, columns=['promocao'], prefix='promocao')
            
        if 'dia_da_semana' in new_sale_df.columns:
            new_sale_df = pd.get_dummies(new_sale_df, columns=['dia_da_semana'], prefix='dia_da_semana')
        
        # Garante que as colunas do novo dado correspondam exatamente às do modelo treinado
        for col in self.columns:
            if col not in new_sale_df.columns:
                new_sale_df[col] = 0
        
        # Remover colunas extras que não são usadas pelo modelo
        extra_cols = [col for col in new_sale_df.columns if col not in self.columns]
        if extra_cols:
            new_sale_df = new_sale_df.drop(columns=extra_cols)
            
        # Garantir que o DataFrame tenha as colunas na mesma ordem do modelo
        new_sale_df = new_sale_df[self.columns]
        
        # Converter todos os tipos para numéricos para garantir compatibilidade
        for col in new_sale_df.columns:
            new_sale_df[col] = pd.to_numeric(new_sale_df[col], errors='coerce')
            
        # Substituir valores NaN por 0
        new_sale_df = new_sale_df.fillna(0)
        
        # Verificar e substituir infinitos usando um método seguro
        for col in new_sale_df.columns:
            try:
                # Tenta identificar valores infinitos com segurança
                mask = ~np.isfinite(new_sale_df[col].values)
                if mask.any():
                    new_sale_df.loc[:, col] = new_sale_df[col].replace([np.inf, -np.inf], 999999)
            except TypeError:
                # Se o tipo não suporta isfinite, converta para float
                new_sale_df[col] = new_sale_df[col].astype(float)
        
        try:
            prediction = self.model.predict(new_sale_df)
            return int(max(0, prediction[0]))  # Garantir que é positivo
        except Exception as e:
            # Em caso de erro, retornar um valor padrão
            print(f"Erro na previsão: {e}")
            return 10  # Valor padrão em caso de erro