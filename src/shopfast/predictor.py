import pandas as pd
import numpy as np

class Predictor:
    """Usa um modelo treinado para fazer novas previsões."""
    def __init__(self, model, columns, preprocessor=None):
        self.model = model
        self.columns = columns
        self.preprocessor = preprocessor  # Opcional: para usar os mesmos encoders

    def predict_demand(self, new_sale_data):
        """Prevê a demanda para um novo conjunto de dados de venda."""
        # Cria um DataFrame a partir dos novos dados
        if isinstance(new_sale_data, dict):
            new_sale_df = pd.DataFrame([new_sale_data])
        elif isinstance(new_sale_data, list):
            new_sale_df = pd.DataFrame(new_sale_data)
        else:
            new_sale_df = new_sale_data.copy()
        
        # Aplica o preprocessamento se o preprocessor estiver disponível
        if self.preprocessor:
            new_sale_df = self.preprocessor.encode_new_data(new_sale_df)
        
        # Processar a coluna 'mes' se ela não existir, mas 'data' existir
        if 'mes' not in new_sale_df.columns and 'data' in new_sale_df.columns:
            try:
                new_sale_df['data'] = pd.to_datetime(new_sale_df['data'])
                new_sale_df['mes'] = new_sale_df['data'].dt.month
                # Adicionar outras features temporais consistentes com o treinamento
                if 'dia' in self.columns:
                    new_sale_df['dia'] = new_sale_df['data'].dt.day
                if 'trimestre' in self.columns:
                    new_sale_df['trimestre'] = new_sale_df['data'].dt.quarter
            except:
                # Se falhar, use um valor padrão
                new_sale_df['mes'] = 1
        
        # Lista de possíveis colunas categóricas para one-hot encoding
        cat_columns = ['categoria', 'promocao', 'dia_da_semana', 'id_produto']
        
        # Aplica o one-hot encoding nas colunas categóricas presentes
        for col in cat_columns:
            if col in new_sale_df.columns:
                new_sale_df = pd.get_dummies(new_sale_df, columns=[col], prefix=col)
        
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
            # Realizar a previsão
            predictions = self.model.predict(new_sale_df)
            
            # Se estamos prevendo para múltiplos itens, retornar todos os valores
            if len(predictions) > 1:
                return [int(max(0, p)) for p in predictions]  # Garante valores positivos
            else:
                return int(max(0, predictions[0]))  # Apenas um valor
                
        except Exception as e:
            # Em caso de erro, retornar um valor padrão
            print(f"Erro na previsão: {e}")
            if len(new_sale_df) > 1:
                return [10] * len(new_sale_df)  # Valores padrão para múltiplos itens
            else:
                return 10  # Valor padrão em caso de erro