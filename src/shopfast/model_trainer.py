"""
Módulo para treinar modelos de previsão de demanda por produto.
Este script realiza o treinamento, avaliação e salvamento do modelo.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import GridSearchCV
from .data_preprocessor import DataPreprocessor
from .demand_model import DemandPredictionModel

class ModelTrainer:
    """Classe que gerencia o treinamento completo do modelo de previsão."""
    
    def __init__(self, dataset_path, output_dir=None):
        """
        Inicializa o treinador de modelo.
        
        Args:
            dataset_path: Caminho para o dataset de treinamento
            output_dir: Diretório onde salvar o modelo (opcional)
        """
        self.dataset_path = dataset_path
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.output_dir = output_dir
            
        # Inicializa variáveis que serão definidas durante o treinamento
        self.preprocessor = None
        self.model = None
        self.feature_columns = None
        self.target_column = 'quantidade_vendida'  # Coluna padrão de demanda
        self.model_columns = None
        
    def carregar_dados(self):
        """Carrega e faz a limpeza inicial dos dados."""
        try:
            # Tenta carregar o arquivo como CSV
            df = pd.read_csv(self.dataset_path, delimiter=',')
            print(f"Dados carregados com sucesso: {len(df)} registros.")
            return df
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            try:
                # Tenta carregar como texto delimitado
                df = pd.read_csv(self.dataset_path, delimiter='\t')
                print(f"Dados carregados com sucesso: {len(df)} registros.")
                return df
            except Exception as e2:
                print(f"Erro ao carregar dados como texto delimitado: {e2}")
                return None
            
    def definir_features_target(self, df, target_column=None):
        """Define as colunas de features e target para o modelo."""
        if target_column:
            self.target_column = target_column
        
        # Remove colunas que não devem ser usadas como features
        colunas_para_remover = ['data']  # A data será transformada em features temporais
        
        # Define as colunas de features (todas exceto a target e as para remover)
        self.feature_columns = [col for col in df.columns 
                               if col != self.target_column and col not in colunas_para_remover]
        
        print(f"Target definido: {self.target_column}")
        print(f"Features selecionadas: {len(self.feature_columns)} colunas")
        return self.feature_columns, self.target_column
        
    def treinar_modelo(self, modelo_tipo="random_forest", otimizar_hiperparametros=False):
        """Treina o modelo com os dados fornecidos."""
        # Carrega os dados
        df = self.carregar_dados()
        if df is None:
            return False
            
        # Define features e target
        self.definir_features_target(df)
        
        # Inicializa o preprocessador
        self.preprocessor = DataPreprocessor(df)
        
        # Preprocessa os dados
        X_encoded, y = self.preprocessor.preprocess(self.feature_columns, self.target_column)
        
        # Salva as colunas do modelo após o encoding
        self.model_columns = X_encoded.columns
        
        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X_encoded, y)
        
        # Se otimizar hiperparâmetros
        if otimizar_hiperparametros:
            print("\nRealizando otimização de hiperparâmetros...")
            best_params = self._otimizar_hiperparametros(X_train, y_train)
            print(f"Melhores parâmetros encontrados: {best_params}")
            
            # Cria o modelo com os melhores parâmetros
            if modelo_tipo == "decision_tree":
                self.model = DemandPredictionModel(
                    model_type=modelo_tipo, 
                    max_depth=best_params.get('max_depth', 10),
                    random_state=42
                )
            elif modelo_tipo == "random_forest":
                self.model = DemandPredictionModel(
                    model_type=modelo_tipo,
                    max_depth=best_params.get('max_depth', 10),
                    random_state=42
                )
            else:
                self.model = DemandPredictionModel(
                    model_type=modelo_tipo,
                    max_depth=best_params.get('max_depth', 10),
                    random_state=42
                )
        else:
            # Cria o modelo com parâmetros padrão
            self.model = DemandPredictionModel(model_type=modelo_tipo)
            
        # Treina o modelo
        self.model.train(X_train, y_train)
        
        # Avalia o modelo
        metricas = self.model.evaluate(X_test, y_test)
        
        # Plota a importância das features
        self.model.plot_feature_importance()
        
        return True
        
    def _otimizar_hiperparametros(self, X_train, y_train):
        """Realiza a otimização de hiperparâmetros usando GridSearchCV."""
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'n_estimators': [50, 100, 200]
        }
        
        # Cria um modelo básico
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(random_state=42)
        
        # Realiza a busca em grade
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_absolute_error'
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
        
    def salvar_modelo(self):
        """Salva o modelo treinado e suas configurações."""
        if self.model is None:
            print("Erro: Nenhum modelo foi treinado para salvar.")
            return False
            
        try:
            # Salva o modelo
            joblib.dump(self.model.model, os.path.join(self.output_dir, 'modelo_demanda.pkl'))
            
            # Salva as colunas do modelo
            joblib.dump(self.model_columns, os.path.join(self.output_dir, 'colunas_modelo.pkl'))
            
            # Salva o preprocessador
            joblib.dump(self.preprocessor, os.path.join(self.output_dir, 'preprocessor_modelo.pkl'))
            
            print(f"\nModelo e configurações salvas com sucesso em: {self.output_dir}")
            return True
        except Exception as e:
            print(f"Erro ao salvar o modelo: {e}")
            return False
