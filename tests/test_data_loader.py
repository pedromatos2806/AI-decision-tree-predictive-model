import unittest
import pandas as pd
import numpy as np
import os
import sys

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'shopfast'))

from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Testes para o DataLoader."""
    
    def setUp(self):
        """Configuração para os testes."""
        # Criar um arquivo temporário para teste
        self.test_file = "test_data.csv"
        self.create_test_data()
        self.data_loader = DataLoader(file_path=self.test_file)
        
    def tearDown(self):
        """Limpeza após os testes."""
        # Remover arquivo temporário
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def create_test_data(self):
        """Cria dados de teste."""
        data = {
            'data': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-02'],
            'produto_id': [101, 102, 101, 103],
            'categoria': ['Roupas', 'Eletrônicos', 'Roupas', 'Utensílios'],
            'quantidade_vendida': [20, 15, 18, 30],
            'preco_unitario': [50.00, 1000.00, 50.00, 20.00],
            'promocao': ['Sim', 'Não', 'Não', 'Sim'],
            'temperatura_media': [30.0, 30.0, 28.0, 25.0],
            'humidade_media': [60.0, 55.0, 65.0, 70.0],
            'dia_da_semana': ['Quarta', 'Quarta', 'Quinta', 'Sexta'],
            'feedback_cliente': [5, 4, 5, 3]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.test_file, index=False)
        
    def test_load_data(self):
        """Testa se os dados são carregados corretamente."""
        data = self.data_loader.load_data()
        
        # Verificar se os dados foram carregados
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        
        # Verificar se todas as colunas esperadas estão presentes
        expected_columns = [
            'data', 'produto_id', 'categoria', 'quantidade_vendida',
            'preco_unitario', 'promocao', 'temperatura_media',
            'humidade_media', 'dia_da_semana', 'feedback_cliente'
        ]
        for col in expected_columns:
            self.assertIn(col, data.columns)
            
        # Verificar se o número de linhas está correto
        self.assertEqual(len(data), 4)
        
    def test_preprocess_data(self):
        """Testa o pré-processamento dos dados."""
        self.data_loader.load_data()
        X_train, X_test, y_train, y_test, product_ids_test = self.data_loader.preprocess_data(
            target_column='quantidade_vendida',
            test_size=0.5,
            random_state=42
        )
        
        # Verificar se os conjuntos de dados foram criados
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(product_ids_test)
        
        # Verificar o tamanho dos conjuntos de treino e teste
        self.assertEqual(len(y_train) + len(y_test), 4)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)
        
        # Verificar se produto_id foi removido das features mas mantido separadamente
        self.assertEqual(len(product_ids_test), 2)
        
    def test_get_feature_names(self):
        """Testa a obtenção dos nomes das features."""
        self.data_loader.load_data()
        self.data_loader.preprocess_data(target_column='quantidade_vendida')
        feature_names = self.data_loader.get_feature_names()
        
        # Verificar se os nomes das features foram obtidos
        self.assertIsNotNone(feature_names)
        self.assertGreater(len(feature_names), 0)
        
        # Verificar se alguns nomes esperados estão presentes
        self.assertIn('preco_unitario', feature_names)
        self.assertIn('temperatura_media', feature_names)
        self.assertIn('humidade_media', feature_names)
        self.assertIn('mes', feature_names)
        self.assertIn('dia', feature_names)
        
        # Verificar se produto_id e quantidade_vendida foram removidos
        self.assertNotIn('produto_id', feature_names)
        self.assertNotIn('quantidade_vendida', feature_names)
        
        # Verificar se existem valores categóricos transformados
        categorical_cols = ['categoria', 'dia_da_semana']
        has_categorical = False
        for col in feature_names:
            for cat_col in categorical_cols:
                if cat_col in col:
                    has_categorical = True
                    break
            if has_categorical:
                break
                
        self.assertTrue(has_categorical, "Features categóricas não estão presentes")

if __name__ == '__main__':
    unittest.main()
