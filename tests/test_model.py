import unittest
import pandas as pd
import numpy as np
import os
import sys

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'shopfast'))

from model_builder import ModelBuilder
from model_evaluator import ModelEvaluator

class TestModel(unittest.TestCase):
    """Testes para o ModelBuilder e ModelEvaluator."""
    
    def setUp(self):
        """Configuração para os testes."""
        # Criar dados sintéticos para teste
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100) * 100
        self.X_test = np.random.rand(30, 5)
        self.y_test = np.random.rand(30) * 100
        self.product_ids_test = np.arange(101, 131)
        self.feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
    def test_model_builder_initialization(self):
        """Testa a inicialização do ModelBuilder."""
        # Testar diferentes tipos de modelos
        model_types = ["decision_tree", "random_forest", "gradient_boosting", "unknown_type"]
        for model_type in model_types:
            builder = ModelBuilder(model_type=model_type)
            self.assertIsNotNone(builder.model, f"Falha ao inicializar modelo do tipo {model_type}")
            
    def test_model_training(self):
        """Testa o treinamento do modelo."""
        builder = ModelBuilder(model_type="random_forest")
        model = builder.train(self.X_train, self.y_train, self.feature_names)
        
        # Verificar se o modelo foi treinado
        self.assertIsNotNone(model)
        
        # Verificar se a importância das features foi calculada
        feature_importance = builder.get_feature_importance()
        self.assertIsNotNone(feature_importance)
        self.assertEqual(len(feature_importance), 5)
        
    def test_model_prediction(self):
        """Testa as previsões do modelo."""
        builder = ModelBuilder(model_type="random_forest")
        builder.train(self.X_train, self.y_train, self.feature_names)
        
        # Fazer previsões
        y_pred = builder.predict(self.X_test)
        
        # Verificar se as previsões têm o formato esperado
        self.assertEqual(len(y_pred), len(self.X_test))
        
    def test_model_evaluation(self):
        """Testa a avaliação do modelo."""
        builder = ModelBuilder(model_type="random_forest")
        model = builder.train(self.X_train, self.y_train, self.feature_names)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, self.X_test, self.y_test, self.product_ids_test)
        
        # Verificar se as métricas foram calculadas
        self.assertIsNotNone(metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        
        # Verificar se as métricas estão no intervalo esperado
        self.assertLessEqual(metrics['r2'], 1.0)
        self.assertGreaterEqual(metrics['mae'], 0.0)
        self.assertGreaterEqual(metrics['rmse'], 0.0)
        self.assertGreaterEqual(metrics['mape'], 0.0)
        
        # Verificar se as previsões foram armazenadas
        predictions = evaluator.get_predictions()
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Verificar se os resultados das previsões incluem os IDs dos produtos
        self.assertTrue(hasattr(evaluator, 'prediction_results'))
        self.assertIn('produto_id', evaluator.prediction_results.columns)
        self.assertIn('demanda_real', evaluator.prediction_results.columns)
        self.assertIn('demanda_prevista', evaluator.prediction_results.columns)

if __name__ == '__main__':
    unittest.main()
