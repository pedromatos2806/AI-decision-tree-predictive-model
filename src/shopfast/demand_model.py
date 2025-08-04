from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

class DemandPredictionModel:
    """Gerencia o treinamento e a avaliação do modelo de Árvore de Decisão."""
    def __init__(self, max_depth=10, random_state=42):
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    def train(self, X_train, y_train):
        """Treina o modelo de previsão de demanda."""
        print("Treinando o modelo de Inteligência Artificial...")
        self.model.fit(X_train, y_train)
        print("Modelo treinado com sucesso.")

    def evaluate(self, X_test, y_test):
        """Avalia o desempenho do modelo e exibe as métricas."""
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print("\n--- Avaliação de Desempenho do Modelo ---")
        print(f"Coeficiente de Determinação (R²): {r2:.2f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.2f} unidades")