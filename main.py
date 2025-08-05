import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from matplotlib.patches import Patch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Adicionar diretório src ao path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'shopfast'))

from data_loader import DataLoader
from model_builder import ModelBuilder
from model_evaluator import ModelEvaluator
from visualization import ModelVisualizer

def configurar_pastas():
    """Configura e cria as pastas necessárias para o projeto."""
    # Diretório base
    pasta_base = os.path.dirname(os.path.abspath(__file__))
    
    # Pasta para dados
    pasta_dados = os.path.join(pasta_base, 'dados')
    if not os.path.exists(pasta_dados):
        os.makedirs(pasta_dados)
        
    # Pasta para resultados
    data_hoje = datetime.datetime.now().strftime('%Y_%m_%d')
    pasta_resultados = os.path.join(pasta_base, 'resultados')
    if not os.path.exists(pasta_resultados):
        os.makedirs(pasta_resultados)
    
    pasta_data = os.path.join(pasta_resultados, data_hoje)
    if not os.path.exists(pasta_data):
        os.makedirs(pasta_data)
    
    pasta_graficos = os.path.join(pasta_data, 'graficos')
    if not os.path.exists(pasta_graficos):
        os.makedirs(pasta_graficos)
        
    pasta_modelos = os.path.join(pasta_base, 'modelos')
    if not os.path.exists(pasta_modelos):
        os.makedirs(pasta_modelos)
    
    return {
        'base': pasta_base,
        'dados': pasta_dados,
        'resultados': pasta_data,
        'graficos': pasta_graficos,
        'modelos': pasta_modelos
    }

def carregar_e_processar_dados(caminho_dados, delimiter=','):
    """Carrega e processa os dados para treinamento."""
    print("\n1. Carregando e pré-processando os dados...")
    data_loader = DataLoader(file_path=caminho_dados, delimiter=delimiter)
    
    # Definir colunas obrigatórias mínimas
    colunas_necessarias = [
        'data', 'produto_id', 'categoria', 'quantidade_vendida',
        'preco_unitario', 'temperatura_media', 'humidade_media', 'dia_da_semana'
    ]
    
    # Carregar dados
    data = data_loader.load_data(required_columns=colunas_necessarias)
    
    # Pré-processar dados
    target_column = 'quantidade_vendida'
    X_train, X_test, y_train, y_test, product_ids_test = data_loader.preprocess_data(
        target_column=target_column, 
        test_size=0.2,
        random_state=42
    )
    
    # Obter nomes das features
    feature_names = data_loader.get_feature_names()
    
    return {
        'data_loader': data_loader,
        'data': data,
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'product_ids_test': product_ids_test,
        'feature_names': feature_names,
        'target_column': target_column
    }

def treinar_modelo(dados_processados, model_type="random_forest"):
    """Treina o modelo de previsão de demanda."""
    print("\n2. Construindo e treinando o modelo...")
    model_builder = ModelBuilder(model_type=model_type, max_depth=10, random_state=42)
    
    model = model_builder.train(
        dados_processados['X_train'], 
        dados_processados['y_train'], 
        dados_processados['feature_names']
    )
    
    feature_importance = model_builder.get_feature_importance()
    
    return {
        'model': model,
        'model_builder': model_builder,
        'feature_importance': feature_importance
    }

def avaliar_modelo(modelo, dados_processados):
    """Avalia o desempenho do modelo."""
    print("\n3. Avaliando o desempenho do modelo...")
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate(
        modelo['model'], 
        dados_processados['X_test'], 
        dados_processados['y_test'], 
        dados_processados['product_ids_test']
    )
    
    return {
        'evaluator': evaluator,
        'metrics': metrics,
        'predictions': evaluator.get_predictions(),
        'prediction_results': evaluator.prediction_results
    }

def visualizar_resultados(modelo, avaliacao, dados_processados, pastas):
    """Gera visualizações dos resultados."""
    print("\n4. Gerando visualizações...")
    visualizer = ModelVisualizer(output_dir=pastas['graficos'])
    
    # Gerar e salvar gráficos
    visualizer.plot_feature_importance(modelo['feature_importance'])
    visualizer.plot_actual_vs_predicted(dados_processados['y_test'], avaliacao['predictions'])
    visualizer.plot_prediction_results(avaliacao['prediction_results'])
    visualizer.plot_error_distribution(dados_processados['y_test'], avaliacao['predictions'])
    
    # Salvar previsões
    prediction_file = os.path.join(pastas['resultados'], "previsoes_demanda.csv")
    avaliacao['prediction_results'].to_csv(prediction_file, index=False)
    print(f"\nResultados de previsão salvos em: {prediction_file}")
    
    return visualizer

def salvar_modelo(modelo, dados_processados, pastas):
    """Salva o modelo treinado e metadados para uso futuro."""
    print("\n5. Salvando o modelo treinado...")
    
    # Caminhos para os arquivos
    modelo_path = os.path.join(pastas['modelos'], 'modelo_demanda.pkl')
    colunas_path = os.path.join(pastas['modelos'], 'colunas_modelo.pkl')
    preprocessor_path = os.path.join(pastas['modelos'], 'preprocessor_modelo.pkl')
    
    # Salvar modelo e metadados
    joblib.dump(modelo['model'], modelo_path)
    
    # Obter colunas do modelo
    if hasattr(dados_processados['data_loader'], 'preprocessor'):
        joblib.dump(dados_processados['data_loader'].preprocessor, preprocessor_path)
        print(f"Preprocessador salvo em: {preprocessor_path}")
    
    # Salvar nomes das colunas
    joblib.dump(dados_processados['feature_names'], colunas_path)
    
    print(f"Modelo salvo em: {modelo_path}")
    print(f"Metadados do modelo salvos em: {colunas_path}")
    
    return {
        'modelo_path': modelo_path,
        'colunas_path': colunas_path,
        'preprocessor_path': preprocessor_path
    }

def recomendar_produtos():
    """Recomenda produtos para compra com base nas previsões de demanda para produtos existentes."""
    print("\n6. Analisando produtos existentes para recomendações de compra...")
    
    # Carregar modelo e metadados
    try:
        model = joblib.load('modelos/modelo_demanda.pkl')
        preprocessor = joblib.load('modelos/preprocessor_modelo.pkl')
        
        # Carregar dados originais
        caminho_dados = os.path.join('dados', '2025.1 - Vendas_semestre.txt')
        dados = pd.read_csv(caminho_dados, sep=',')
        
        # Limpar nomes das colunas (remover espaços extras)
        dados.columns = dados.columns.str.strip()
        
        # Filtrar apenas linhas com datas válidas
        dados['data'] = pd.to_datetime(dados['data'], errors='coerce')
        dados = dados.dropna(subset=['data'])
        
        # Pré-processar para previsão
        # Adicionar features de data
        dados['mes'] = dados['data'].dt.month
        dados['dia'] = dados['data'].dt.day
        
        # Converter promocao para binário
        dados['promocao'] = dados['promocao'].apply(lambda x: 1 if x in ['Sim', 'SIM', 'sim', 'S', 's', '1', 'True', 'true'] else 0)
        
        # Agrupar dados por produto para análise
        produtos_dados = dados.groupby('produto_id').agg({
            'categoria': 'first',  # Pegar a primeira categoria
            'quantidade_vendida': ['mean', 'sum', 'count'],  # Estatísticas de vendas
            'preco_unitario': 'mean',  # Preço médio
            'promocao': 'mean',  # Percentual de dias com promoção
            'temperatura_media': 'mean',
            'humidade_media': 'mean',
            'feedback_cliente': 'mean',
        }).reset_index()
        
        # Renomear colunas
        produtos_dados.columns = ['produto_id', 'categoria', 'venda_media', 'venda_total', 
                                  'dias_venda', 'preco_medio', 'perc_promocao', 
                                  'temperatura_media', 'humidade_media', 'feedback_medio']
        
        # Preparar dados para previsão
        # Obter as últimas tendências para cada produto
        tendencias = dados.sort_values('data').groupby('produto_id').tail(5)
        
        # Para cada produto, obter os valores mais recentes para prever a próxima demanda
        produtos_para_previsao = []
        for produto_id in produtos_dados['produto_id'].unique():
            ultimos_dados = tendencias[tendencias['produto_id'] == produto_id].iloc[-1:].copy()
            
            if not ultimos_dados.empty:
                ultimos_dados['data_previsao'] = pd.to_datetime('today')
                ultimos_dados['mes'] = ultimos_dados['data_previsao'].dt.month
                ultimos_dados['dia'] = ultimos_dados['data_previsao'].dt.day
                produtos_para_previsao.append(ultimos_dados)
        
        if not produtos_para_previsao:
            print("Não foi possível preparar dados para previsão.")
            return None
            
        # Concatenar todos os produtos para previsão
        df_previsao = pd.concat(produtos_para_previsao, ignore_index=True)
        
        # Remover colunas não usadas no modelo e a quantidade vendida (target)
        df_previsao = df_previsao.drop(['quantidade_vendida', 'data', 'data_previsao'], axis=1, errors='ignore')
        
        # Preparar dados para o modelo
        X_pred = preprocessor.transform(df_previsao)
        
        # Fazer previsão
        previsoes = model.predict(X_pred)
        
        # Criar DataFrame de resultados
        resultados = pd.DataFrame({
            'produto_id': df_previsao['produto_id'],
            'categoria': df_previsao['categoria'],
            'preco_unitario': df_previsao['preco_unitario'],
            'feedback_cliente': df_previsao['feedback_cliente'],
            'demanda_prevista': np.round(previsoes).astype(int)
        })
        
        # Adicionar venda média e total para comparação
        resultados = resultados.merge(
            produtos_dados[['produto_id', 'venda_media', 'venda_total', 'dias_venda']], 
            on='produto_id', how='left'
        )
        
        # Ordenar por demanda prevista (decrescente)
        resultados = resultados.sort_values('demanda_prevista', ascending=False)
        
        # Calcular índice de prioridade (demanda prevista / preço)
        resultados['prioridade_compra'] = resultados['demanda_prevista'] / resultados['preco_unitario']
        
        # Classificar produtos para recomendação
        resultados['recomendacao'] = 'Normal'
        resultados.loc[resultados['demanda_prevista'] > resultados['venda_media'] * 1.2, 'recomendacao'] = 'Alta'
        resultados.loc[resultados['demanda_prevista'] < resultados['venda_media'] * 0.8, 'recomendacao'] = 'Baixa'
        
        # Exibir resultados
        print("\nRecomendações de Produtos para Compra (Top 10 por demanda prevista):")
        cols_exibir = ['produto_id', 'categoria', 'preco_unitario', 'venda_media', 
                       'demanda_prevista', 'recomendacao']
        print(resultados[cols_exibir].head(10).to_string(index=False))
        
        return resultados
    
    except FileNotFoundError:
        print("Modelo não encontrado. Execute o treinamento primeiro.")
        return None
    except Exception as e:
        print(f"Erro ao recomendar produtos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def criar_grafico_recomendacao(resultados, pasta_graficos):
    """Cria um gráfico com as recomendações de produtos."""
    if resultados is None:
        return
    
    # Selecionar os top 15 produtos para o gráfico
    top_produtos = resultados.head(15)
    
    plt.figure(figsize=(12, 8))
    
    # Configurar barras
    cores = {'Alta': '#2ca02c', 'Normal': '#1f77b4', 'Baixa': '#d62728'}
    cores_barras = [cores[rec] for rec in top_produtos['recomendacao']]
    
    # Criar gráfico
    barras = plt.bar(
        top_produtos['produto_id'].astype(str), 
        top_produtos['demanda_prevista'],
        color=cores_barras
    )
    
    # Adicionar valores nas barras
    for i, barra in enumerate(barras):
        altura = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width()/2, 
            altura + 0.5,
            str(int(top_produtos['demanda_prevista'].iloc[i])),
            ha='center', va='bottom'
        )
    
    # Adicionar linha para média histórica
    plt.plot(
        range(len(top_produtos)), 
        top_produtos['venda_media'], 
        'o--', color='#ff7f0e', 
        label='Média Histórica'
    )
    
    # Configurar gráfico
    plt.title('Recomendação de Produtos para Compra', fontsize=15)
    plt.xlabel('ID do Produto')
    plt.ylabel('Unidades')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Adicionar legenda para cores
    legend_elements = [
        Patch(facecolor=cores['Alta'], label='Prioridade Alta'),
        Patch(facecolor=cores['Normal'], label='Prioridade Normal'),
        Patch(facecolor=cores['Baixa'], label='Prioridade Baixa'),
    ]
    plt.legend(handles=legend_elements, title='Recomendação')
    
    # Salvar gráfico
    data_hoje = datetime.datetime.now().strftime('%Y%m%d')
    plt.tight_layout()
    caminho_grafico = os.path.join(pasta_graficos, f'recomendacao_produtos_{data_hoje}.png')
    plt.savefig(caminho_grafico, dpi=300)
    print(f"\nGráfico de recomendações salvo em: {caminho_grafico}")
    
    return caminho_grafico

def main(caminho_dados_alternativo=None):
    """Função principal que executa o fluxo de previsão de demanda."""
    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA - ShopFast")
    print("=" * 70)
    
    # Configurar pastas
    pastas = configurar_pastas()
    
    # Configurar caminho para o arquivo de dados (com suporte a parâmetro)
    if caminho_dados_alternativo:
        caminho_dados = caminho_dados_alternativo
        print(f"Usando arquivo de dados fornecido: {caminho_dados}")
    else:
        caminho_dados = os.path.join(pastas['dados'], '2025.1 - Vendas_semestre.txt')
    
    try:
        # 1. Carregar e processar os dados
        dados_processados = carregar_e_processar_dados(caminho_dados, delimiter=',')
        
        # 2. Treinar o modelo
        modelo_treinado = treinar_modelo(dados_processados, model_type="random_forest")
        
        # 3. Avaliar o modelo
        avaliacao = avaliar_modelo(modelo_treinado, dados_processados)
        
        # 4. Visualizar resultados
        visualizer = visualizar_resultados(modelo_treinado, avaliacao, dados_processados, pastas)
        
        # 5. Salvar o modelo
        modelo_salvo = salvar_modelo(modelo_treinado, dados_processados, pastas)
        
        # 6. Recomendar produtos existentes para compra (em vez de prever novos produtos)
        resultados_recomendacao = recomendar_produtos()
        
        # 7. Criar gráfico para produtos recomendados
        if resultados_recomendacao is not None:
            caminho_grafico = criar_grafico_recomendacao(resultados_recomendacao, pastas['graficos'])
        
        # Mostrar resumo final
        print("\n" + "=" * 70)
        print("ANÁLISE DE PREVISÃO DE DEMANDA CONCLUÍDA")
        print(f"Acurácia do modelo (R²): {avaliacao['metrics']['r2']:.4f}")
        print(f"Erro percentual médio: {avaliacao['metrics']['mape']:.2f}%")
        print(f"Erro médio absoluto: {avaliacao['metrics']['mae']:.2f} unidades")
        print("=" * 70)
        
        # Mostrar gráficos
        plt.show()
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nVerificando o arquivo de entrada para diagnóstico...")
        try:
            # Tentar ler as primeiras linhas para diagnóstico
            with open(caminho_dados, 'r', encoding='utf-8', errors='ignore') as f:
                primeiras_linhas = [next(f) for _ in range(5)]
                print("\nPrimeiras 5 linhas do arquivo:")
                for linha in primeiras_linhas:
                    print(linha.strip())
        except Exception as e2:
            print(f"Não foi possível ler o arquivo para diagnóstico: {str(e2)}")
            
        # Script para ler o arquivo de texto
        try:
            arquivo_entrada = caminho_dados_alternativo if 'caminho_dados_alternativo' in locals() else "C:\\Users\\pedro\\Downloads\\nova-base.txt"
            with open(arquivo_entrada, "r", encoding='utf-8', errors='ignore') as arquivo:
                conteudo = arquivo.read()
                print(conteudo)  # ou processe o conteúdo como necessário

            print(f"Arquivo {arquivo_entrada} processado com sucesso!")
        except Exception as e3:
            print(f"Erro ao processar o arquivo de texto: {str(e3)}")
            
if __name__ == "__main__":
    # Verificar se foi passado um caminho como argumento de linha de comando
    if len(sys.argv) > 1:
        caminho_arquivo = sys.argv[1]
        print(f"Usando caminho fornecido via linha de comando: {caminho_arquivo}")
        main(caminho_arquivo)
    else:
        main()