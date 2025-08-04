from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DemandPredictionModel:
    """Gerencia o treinamento e a avaliação do modelo de previsão de demanda."""
    def __init__(self, model_type="random_forest", max_depth=10, random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        
        # Seleção do modelo com base no tipo especificado
        if model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=max_depth,
                random_state=random_state
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            print(f"Tipo de modelo '{model_type}' não reconhecido. Usando Random Forest.")
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=max_depth,
                random_state=random_state
            )
            
        print(f"Modelo inicializado: {type(self.model).__name__}")

    def train(self, X_train, y_train):
        """Treina o modelo de previsão de demanda."""
        print(f"Treinando o modelo de {type(self.model).__name__}...")
        print(f"Dimensões dos dados de treino: X={X_train.shape}, y={y_train.shape}")
        self.model.fit(X_train, y_train)
        print("Modelo treinado com sucesso.")
        
        # Se for um modelo baseado em árvores, mostrar a importância das features
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nImportância das 10 principais features:")
            print(self.feature_importance.head(10))

    def evaluate(self, X_test, y_test, product_ids=None):
        """Avalia o desempenho do modelo e exibe as métricas."""
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("\n--- Avaliação de Desempenho do Modelo ---")
        print(f"Coeficiente de Determinação (R²): {r2:.2f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.2f} unidades")
        print(f"Erro Quadrático Médio (RMSE): {rmse:.2f} unidades")
        
        # Exibe as previsões por ID de produto
        if product_ids is not None:
            print("\n--- Previsões de Demanda por Produto ---")
            results = pd.DataFrame({
                'id_produto': product_ids,
                'demanda_real': y_test,
                'demanda_prevista': np.round(y_pred, 0).astype(int)
            })
            print(results.to_string(index=False))
        
        # Retorna as métricas e previsões para uso posterior
        results_dict = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
        }
        
        if product_ids is not None:
            results_dict['previsoes'] = pd.DataFrame({
                'id_produto': product_ids,
                'demanda_prevista': np.round(y_pred, 0).astype(int)
            })
        
        return results_dict
        
    def plot_feature_importance(self, top_n=20):
        """Plota um gráfico com as features mais importantes."""
        if hasattr(self, 'feature_importance'):
            plt.figure(figsize=(12, 8))
            
            # Seleciona as top_n features mais importantes
            top_features = self.feature_importance.head(top_n)
            
            # Cria o gráfico de barras horizontais
            plt.barh(
                range(len(top_features)), 
                top_features['Importance'],
                align='center'
            )
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.title('Importância das Features na Previsão de Demanda', fontsize=15)
            plt.xlabel('Importância')
            plt.tight_layout()
            
            return plt.gcf()  # Retorna a figura para salvar ou mostrar
        else:
            print("Importância das features não disponível para este modelo.")
            return None