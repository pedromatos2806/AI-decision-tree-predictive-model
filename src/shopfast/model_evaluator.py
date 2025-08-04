import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ModelEvaluator:
    """Classe para avaliar o desempenho de modelos de previsão de demanda."""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = None
        
    def evaluate(self, model, X_test, y_test, product_ids=None):
        """Avalia o modelo e retorna métricas de desempenho."""
        # Fazer previsões
        y_pred = model.predict(X_test)
        self.predictions = y_pred
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(1, y_test))) * 100
        
        # Armazenar métricas
        self.metrics = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        # Exibir resultados
        print("\n--- Avaliação de Desempenho do Modelo ---")
        print(f"Coeficiente de Determinação (R²): {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.2f} unidades")
        print(f"Erro Quadrático Médio (RMSE): {rmse:.2f} unidades")
        print(f"Erro Percentual Absoluto Médio (MAPE): {mape:.2f}%")
        
        # Se IDs de produtos foram fornecidos, exibir previsões por produto
        if product_ids is not None:
            prediction_df = pd.DataFrame({
                'produto_id': product_ids,
                'demanda_real': y_test,
                'demanda_prevista': np.round(y_pred, 0).astype(int)
            })
            
            print("\n--- Previsões por Produto (Primeiros 10) ---")
            print(prediction_df.head(10).to_string(index=False))
            
            # Armazenar resultados completos
            self.prediction_results = prediction_df
            
        return self.metrics
        
    def get_metrics(self):
        """Retorna as métricas calculadas."""
        if not self.metrics:
            raise ValueError("Métricas não disponíveis. Execute evaluate() primeiro.")
        return self.metrics
        
    def get_predictions(self):
        """Retorna as previsões."""
        if self.predictions is None:
            raise ValueError("Previsões não disponíveis. Execute evaluate() primeiro.")
        return self.predictions
