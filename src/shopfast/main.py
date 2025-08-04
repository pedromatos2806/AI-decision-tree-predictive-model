import os
import pandas as pd
import numpy as np
from data_loader import DataLoader
from model_builder import ModelBuilder
from model_evaluator import ModelEvaluator
from visualization import ModelVisualizer

def main(data_file, output_dir=None, model_type="random_forest", test_size=0.2, random_state=42):
    """
    Função principal que executa o fluxo completo de previsão de demanda.
    
    Args:
        data_file: Caminho para o arquivo de dados
        output_dir: Diretório para salvar resultados
        model_type: Tipo de modelo a ser usado ("decision_tree", "random_forest" ou "gradient_boosting")
        test_size: Proporção do conjunto de teste
        random_state: Semente para reprodutibilidade
    """
    print("=" * 50)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("=" * 50)
    
    # Criar diretório de saída se não existir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório de saída criado: {output_dir}")
    
    # 1. Carregar e pré-processar os dados
    print("\n1. Carregando e pré-processando os dados...")
    data_loader = DataLoader(file_path=data_file)
    data = data_loader.load_data()
    
    # Renomear 'quantidade_vendida' para manter consistência com a nomenclatura
    target_column = 'quantidade_vendida'
    
    X_train, X_test, y_train, y_test, product_ids_test = data_loader.preprocess_data(
        target_column=target_column, 
        test_size=test_size,
        random_state=random_state
    )
    
    # Obter nomes das features após transformação
    feature_names = data_loader.get_feature_names()
    
    # 2. Construir e treinar o modelo
    print("\n2. Construindo e treinando o modelo...")
    model_builder = ModelBuilder(
        model_type=model_type,
        max_depth=10,
        random_state=random_state
    )
    
    model = model_builder.train(X_train, y_train, feature_names)
    feature_importance = model_builder.get_feature_importance()
    
    # 3. Avaliar o modelo
    print("\n3. Avaliando o desempenho do modelo...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test, product_ids_test)
    
    # 4. Visualizar resultados
    print("\n4. Gerando visualizações...")
    visualizer = ModelVisualizer(output_dir=output_dir)
    
    print("Gerando gráfico de importância das features...")
    feature_importance_plot = visualizer.plot_feature_importance(feature_importance)
    
    print("Gerando gráfico de valores reais vs. previstos...")
    actual_vs_predicted_plot = visualizer.plot_actual_vs_predicted(y_test, evaluator.get_predictions())
    
    print("Gerando gráfico de resultados de previsão...")
    prediction_results_plot = visualizer.plot_prediction_results(evaluator.prediction_results)
    
    print("Gerando gráfico de distribuição de erros...")
    error_distribution_plot = visualizer.plot_error_distribution(y_test, evaluator.get_predictions())
    
    # 5. Salvar resultados de previsão
    if output_dir:
        prediction_file = os.path.join(output_dir, "previsoes_demanda.csv")
        evaluator.prediction_results.to_csv(prediction_file, index=False)
        print(f"\nResultados de previsão salvos em: {prediction_file}")
    
    # Mostrar os gráficos
    plt.show()
    
    print("\n" + "=" * 50)
    print("Análise de Previsão de Demanda Concluída")
    print(f"Acurácia do modelo (R²): {metrics['r2']:.4f}")
    print(f"Erro percentual médio: {metrics['mape']:.2f}%")
    print("=" * 50)
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': evaluator.prediction_results
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configurar parâmetros
    DATA_FILE = r"c:\Users\pedro\OneDrive\Documentos\Projetos\IA\dados\2025.1 - Vendas_semestre.txt"
    OUTPUT_DIR = r"c:\Users\pedro\OneDrive\Documentos\Projetos\IA\resultados\previsao_demanda"
    
    # Executar análise
    results = main(DATA_FILE, OUTPUT_DIR, model_type="random_forest")
