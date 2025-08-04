from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd

class ModelBuilder:
    """Classe para criar e treinar modelos de previsão de demanda."""
    
    def __init__(self, model_type="random_forest", max_depth=10, random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.max_depth = max_depth
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
        # Inicializar modelo
        self._initialize_model()
        
    def _initialize_model(self):
        """Inicializa o modelo de acordo com o tipo especificado."""
        if self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                random_state=self.random_state
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            print(f"Tipo de modelo '{self.model_type}' não reconhecido. Usando Random Forest.")
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            
        print(f"Modelo inicializado: {type(self.model).__name__}")
        
    def train(self, X_train, y_train, feature_names=None):
        """Treina o modelo de previsão de demanda."""
        if self.model is None:
            self._initialize_model()
            
        self.feature_names = feature_names
        
        print(f"Treinando o modelo de {type(self.model).__name__}...")
        print(f"Dimensões dos dados de treino: X={X_train.shape}, y={y_train.shape}")
        
        self.model.fit(X_train, y_train)
        print("Modelo treinado com sucesso.")
        
        # Calcular importância das features se disponível
        if hasattr(self.model, 'feature_importances_'):
            # Se nomes de features não foram fornecidos, usar índices
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nImportância das 10 principais features:")
            print(self.feature_importance.head(10))
            
        return self.model
        
    def predict(self, X_test):
        """Faz previsões usando o modelo treinado."""
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
            
        return self.model.predict(X_test)
        
    def get_feature_importance(self):
        """Retorna a importância das features."""
        if self.feature_importance is None:
            raise ValueError("Importância das features não disponível. O modelo precisa ser treinado primeiro.")
            
        return self.feature_importance
