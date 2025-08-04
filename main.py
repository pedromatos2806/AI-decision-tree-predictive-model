# main.py (Versão Final para Treinamento)

import joblib
from src.shopfast.data_loader import DataLoader
from src.shopfast.data_preprocessor import DataPreprocessor
from src.shopfast.demand_model import DemandPredictionModel
from src.shopfast.predictor import Predictor

def main():
    caminho_dados = 'dados/2025.1 - Vendas_semestre.txt'
    
    # 1. Carregar Dados (agora com o delimitador correto)
    data_loader = DataLoader(caminho_dados, delimiter=',')
    df = data_loader.load_and_clean_data()

    if df is not None:
        # 2. Pré-processar Dados
        preprocessor = DataPreprocessor(df)
        feature_cols = ['preco_unitario', 'temperatura_media', 'humidade_media', 'feedback_cliente', 'mes', 'categoria', 'promocao', 'dia_da_semana']
        target_col = 'quantidade_vendida'
        X_encoded, y = preprocessor.preprocess(feature_cols, target_col)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_encoded, y)

        # 3. Treinar e Avaliar Modelo
        model_manager = DemandPredictionModel()
        model_manager.train(X_train, y_train)
        model_manager.evaluate(X_test, y_test)

        # 4. SALVAR O MODELO E AS COLUNAS PARA USO FUTURO
        print("\nSalvando o modelo treinado...")
        joblib.dump(model_manager.model, 'modelo_demanda.pkl')
        joblib.dump(preprocessor.encoder_columns, 'colunas_modelo.pkl')
        print("Modelo salvo com sucesso como 'modelo_demanda.pkl'")

if __name__ == '__main__':
    main()