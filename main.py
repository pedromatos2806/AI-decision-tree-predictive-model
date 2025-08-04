import os
import sys
import pandas as pd

# Adicionar diretório src ao path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'shopfast'))

from data_loader import DataLoader
from model_builder import ModelBuilder
from model_evaluator import ModelEvaluator
from visualization import ModelVisualizer
import matplotlib.pyplot as plt

def main():
    """Função principal que executa o fluxo de previsão de demanda."""
    print("=" * 50)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("=" * 50)
    
    # Configurar caminhos
    caminho_dados = os.path.join(os.path.dirname(__file__), 'dados', '2025.1 - Vendas_semestre.txt')
    caminho_resultados = os.path.join(os.path.dirname(__file__), 'resultados', 'previsao_demanda')
    
    # Criar diretório de resultados se não existir
    if not os.path.exists(caminho_resultados):
        os.makedirs(caminho_resultados)
        print(f"Diretório de resultados criado: {caminho_resultados}")
    
    # 1. Carregar dados com validação mais flexível
    print("\n1. Carregando e pré-processando os dados...")
    data_loader = DataLoader(file_path=caminho_dados, delimiter=',')
    
    # Definir colunas obrigatórias mínimas
    colunas_necessarias = [
        'data', 'produto_id', 'categoria', 'quantidade_vendida',
        'preco_unitario', 'temperatura_media', 'humidade_media', 'dia_da_semana'
    ]
    
    try:
        data = data_loader.load_data(required_columns=colunas_necessarias)
        
        # 2. Pré-processar dados
        target_column = 'quantidade_vendida'
        X_train, X_test, y_train, y_test, product_ids_test = data_loader.preprocess_data(
            target_column=target_column, 
            test_size=0.2,
            random_state=42
        )
        
        # Obter nomes das features
        feature_names = data_loader.get_feature_names()
        
        # 3. Treinar modelo
        print("\n2. Construindo e treinando o modelo...")
        model_builder = ModelBuilder(model_type="random_forest", max_depth=10, random_state=42)
        model = model_builder.train(X_train, y_train, feature_names)
        feature_importance = model_builder.get_feature_importance()
        
        # 4. Avaliar modelo
        print("\n3. Avaliando o desempenho do modelo...")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test, y_test, product_ids_test)
        
        # 5. Visualizar resultados
        print("\n4. Gerando visualizações...")
        visualizer = ModelVisualizer(output_dir=caminho_resultados)
        
        # Gerar e salvar gráficos
        visualizer.plot_feature_importance(feature_importance)
        visualizer.plot_actual_vs_predicted(y_test, evaluator.get_predictions())
        visualizer.plot_prediction_results(evaluator.prediction_results)
        visualizer.plot_error_distribution(y_test, evaluator.get_predictions())
        
        # Salvar previsões
        prediction_file = os.path.join(caminho_resultados, "previsoes_demanda.csv")
        evaluator.prediction_results.to_csv(prediction_file, index=False)
        print(f"\nResultados de previsão salvos em: {prediction_file}")
        
        # Mostrar resumo final
        print("\n" + "=" * 50)
        print("Análise de Previsão de Demanda Concluída")
        print(f"Acurácia do modelo (R²): {metrics['r2']:.4f}")
        print(f"Erro percentual médio: {metrics['mape']:.2f}%")
        print("=" * 50)
        
        # Mostrar gráficos
        plt.show()
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nVerificando o arquivo de entrada para diagnóstico...")
        try:
            # Tentar ler as primeiras linhas para diagnóstico
            with open(caminho_dados, 'r') as f:
                primeiras_linhas = [next(f) for _ in range(5)]
                print("\nPrimeiras 5 linhas do arquivo:")
                for linha in primeiras_linhas:
                    print(linha.strip())
        except Exception as e2:
            print(f"Não foi possível ler o arquivo para diagnóstico: {str(e2)}")

if __name__ == "__main__":
    main()