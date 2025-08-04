"""
Análise de Demanda Completa - ShopFast

Este script realiza um fluxo completo de análise de demanda:
1. Carrega o modelo treinado
2. Realiza previsões para produtos específicos usando a data atual
3. Gera visualizações da previsão de demanda
4. Analisa o dataset histórico completo
5. Salva os resultados em arquivos CSV
"""

import pandas as pd
import numpy as np
import joblib
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from src.shopfast.data_loader import DataLoader

# Configurações globais para melhorar a visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def configurar_pasta_saida():
    """Configura a pasta para salvar os arquivos de saída."""
    # Cria pastas para armazenar os resultados se não existirem
    data_hoje = datetime.datetime.now().strftime('%Y_%m_%d')
    pasta_base = os.path.dirname(os.path.abspath(__file__))
    
    pasta_resultados = os.path.join(pasta_base, 'resultados')
    if not os.path.exists(pasta_resultados):
        os.makedirs(pasta_resultados)
    
    pasta_data = os.path.join(pasta_resultados, data_hoje)
    if not os.path.exists(pasta_data):
        os.makedirs(pasta_data)
    
    pasta_graficos = os.path.join(pasta_data, 'graficos')
    if not os.path.exists(pasta_graficos):
        os.makedirs(pasta_graficos)
    
    print(f"\nDiretório de resultados criado: {pasta_data}")
    return pasta_data, pasta_graficos


def carregar_modelo():
    """Carrega o modelo treinado e as colunas."""
    try:
        model = joblib.load('modelo_demanda.pkl')
        model_columns = joblib.load('colunas_modelo.pkl')
        print("Modelo 'modelo_demanda.pkl' carregado com sucesso.")
        return model, model_columns
    except FileNotFoundError:
        print("ERRO: Arquivos de modelo não encontrados.")
        print("Rode o script 'main.py' primeiro para treinar e salvar o modelo.")
        return None, None


def prever_novos_produtos(model, model_columns, produtos):
    """Faz previsões para novos produtos com a data atual."""
    if model is None or model_columns is None:
        return None
    
    # Aplica a data atual em todas as entradas
    data_atual = datetime.datetime.now().strftime('%Y-%m-%d')
    for produto in produtos:
        produto['data'] = data_atual
    
    # Converte os dados para um DataFrame
    df_produtos = pd.DataFrame(produtos)
    
    # Adiciona a coluna 'mes' a partir da data
    df_produtos['mes'] = pd.to_datetime(df_produtos['data']).dt.month
    
    # Aplica o one-hot encoding
    df_encoded = pd.get_dummies(df_produtos)
    
    # Reorganiza as colunas para o modelo
    df_final = df_encoded.reindex(columns=model_columns, fill_value=False)
    
    # Faz a previsão
    previsoes = model.predict(df_final)
    
    # Adiciona as previsões ao DataFrame
    df_produtos['demanda_prevista'] = previsoes.astype(int)
    
    return df_produtos, data_atual


def gerar_grafico_barras(resultados, data_atual, pasta_graficos):
    """Gera um gráfico de barras para as previsões de demanda por categoria."""
    plt.figure(figsize=(12, 8))
    
    # Dados para o gráfico
    categorias = resultados['categoria']
    demandas = resultados['demanda_prevista']
    precos = resultados['preco_unitario']
    
    # Cores por categoria
    cores = {'Eletrônicos': 'blue', 'Roupas': 'red', 'Utensílios': 'green'}
    cores_barras = [cores.get(cat, 'gray') for cat in categorias]
    
    # Cria o gráfico de barras
    barras = plt.bar(range(len(categorias)), demandas, color=cores_barras)
    
    # Adiciona rótulos e título
    plt.title(f'Previsão de Demanda por Categoria - {data_atual}', fontsize=16)
    plt.xlabel('Categoria de Produto', fontsize=12)
    plt.ylabel('Demanda Prevista (unidades)', fontsize=12)
    plt.xticks(range(len(categorias)), categorias, rotation=45)
    
    # Adiciona valores nas barras
    for i, bar in enumerate(barras):
        altura = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., altura + 0.5,
                f'{int(demandas.iloc[i])}', 
                ha='center', va='bottom', fontsize=10)
        
        # Adiciona o preço abaixo da categoria
        plt.text(bar.get_x() + bar.get_width()/2., -2.5,
                f'R$ {precos.iloc[i]:.2f}', 
                ha='center', va='top', fontsize=8, color='darkblue')
    
    # Adiciona uma legenda de cores
    legend_elements = [Patch(facecolor=cor, label=cat) for cat, cor in cores.items()]
    plt.legend(handles=legend_elements, title='Categorias', loc='upper right')
    
    # Ajusta o layout e exibe o grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salva o gráfico
    nome_arquivo = f'grafico_demanda_categorias_{data_atual.replace("-", "_")}.png'
    caminho_completo = os.path.join(pasta_graficos, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300)
    
    return caminho_completo


def prever_dataset_historico(model, model_columns, caminho_dados):
    """Processa e faz previsões para o dataset histórico completo."""
    print("\nIniciando análise do dataset histórico completo...")
    
    # Carrega o dataset histórico
    data_loader = DataLoader(caminho_dados, delimiter=',')
    df_original = data_loader.load_and_clean_data()
    
    if df_original is None:
        print(f"ERRO: Dataset original não encontrado ou não pôde ser carregado.")
        return None
    
    # Prepara os dados para previsão
    df_para_prever = df_original.copy()
    df_para_prever['mes'] = pd.to_datetime(df_para_prever['data']).dt.month
    df_encoded = pd.get_dummies(df_para_prever)
    
    # Ajusta as colunas para o modelo
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Filtra apenas as colunas necessárias
    colunas_existentes = [col for col in model_columns if col in df_encoded.columns]
    df_final = df_encoded[colunas_existentes]
    
    # Realiza as previsões
    previsoes = model.predict(df_final)
    df_original['demanda_prevista'] = previsoes.astype(int)
    
    print(f"Dataset histórico processado: {len(df_original)} entradas analisadas.")
    print(f"Estatísticas das previsões: min={df_original['demanda_prevista'].min()}, "
          f"max={df_original['demanda_prevista'].max()}, média={df_original['demanda_prevista'].mean():.2f}")
    
    return df_original


def gerar_grafico_historico(df_historico, pasta_graficos):
    """Gera gráficos para os dados históricos."""
    # 1. Gráfico de linhas para demanda real vs. prevista ao longo do tempo
    plt.figure(figsize=(16, 8))
    
    # Converte data para formato datetime para ordenação adequada
    df_historico['data'] = pd.to_datetime(df_historico['data'])
    df_agrupado = df_historico.groupby(['data', 'categoria']).agg({
        'quantidade_vendida': 'mean',
        'demanda_prevista': 'mean'
    }).reset_index()
    
    # Obtém categorias únicas
    categorias = df_historico['categoria'].unique()
    cores = {'Eletrônicos': 'blue', 'Roupas': 'red', 'Utensílios': 'green'}
    
    # Para cada categoria, plotar duas linhas (real e previsto)
    for categoria in categorias:
        df_cat = df_agrupado[df_agrupado['categoria'] == categoria]
        cor = cores.get(categoria, 'gray')
        
        plt.plot(df_cat['data'], df_cat['quantidade_vendida'], 
                 marker='o', linestyle='-', color=cor, alpha=0.7,
                 label=f'{categoria} (Real)')
        plt.plot(df_cat['data'], df_cat['demanda_prevista'], 
                 marker='x', linestyle='--', color=cor,
                 label=f'{categoria} (Previsto)')
    
    # Formatar eixo x para mostrar datas adequadamente
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gcf().autofmt_xdate()
    
    plt.title('Demanda Real vs. Prevista por Categoria ao Longo do Tempo', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Demanda (unidades)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salva o gráfico
    caminho_grafico = os.path.join(pasta_graficos, 'demanda_historica_tempo.png')
    plt.savefig(caminho_grafico, dpi=300)
    
    # 2. Gráfico de barras agrupadas por categoria
    plt.figure(figsize=(14, 8))
    
    # Calcula médias por categoria
    df_medias = df_historico.groupby('categoria').agg({
        'quantidade_vendida': 'mean',
        'demanda_prevista': 'mean',
        'preco_unitario': 'mean'
    }).reset_index()
    
    # Posições das barras
    x = np.arange(len(df_medias['categoria']))
    width = 0.35
    
    # Cria barras agrupadas
    plt.bar(x - width/2, df_medias['quantidade_vendida'], width, label='Real', color='skyblue')
    plt.bar(x + width/2, df_medias['demanda_prevista'], width, label='Previsto', color='salmon')
    
    # Adiciona rótulos
    plt.title('Comparação de Demanda Real vs. Prevista por Categoria', fontsize=16)
    plt.xlabel('Categoria', fontsize=12)
    plt.ylabel('Demanda Média (unidades)', fontsize=12)
    plt.xticks(x, df_medias['categoria'])
    
    # Adiciona valores nas barras
    for i, v in enumerate(df_medias['quantidade_vendida']):
        plt.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(df_medias['demanda_prevista']):
        plt.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Adiciona preço médio abaixo das categorias
    for i, p in enumerate(df_medias['preco_unitario']):
        plt.text(i, -5, f'R$ {p:.2f}', ha='center', va='bottom', fontsize=8, color='darkblue')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Salva o gráfico
    caminho_grafico2 = os.path.join(pasta_graficos, 'comparacao_categoria_media.png')
    plt.savefig(caminho_grafico2, dpi=300)
    
    return [caminho_grafico, caminho_grafico2]


def exibir_detalhe_previsoes(resultados, data_atual):
    """Exibe os detalhes das previsões no console."""
    print("\n" + "="*80)
    print(f"DETALHES DAS PREVISÕES PARA A DATA ATUAL: {data_atual}")
    print("="*80)
    
    for i, row in resultados.iterrows():
        print(f"\nItem #{i+1}:")
        print(f"  Data: {row['data']} (data atual)")
        print(f"  Categoria: {row['categoria']}")
        print(f"  Preço unitário: R$ {row['preco_unitario']:.2f}")
        print(f"  Promoção: {row['promocao']}")
        print(f"  Temperatura média: {row['temperatura_media']}°C")
        print(f"  Umidade média: {row['humidade_media']}%")
        print(f"  Dia da semana: {row['dia_da_semana']}")
        print(f"  Feedback cliente: {row['feedback_cliente']}")
        print(f"  DEMANDA PREVISTA: {row['demanda_prevista']} unidades")
        print("-"*50)
    
    # Calcula estatísticas para mostrar no resumo
    min_demanda = resultados['demanda_prevista'].min()
    max_demanda = resultados['demanda_prevista'].max()
    media_demanda = resultados['demanda_prevista'].mean()
    
    print("\nRESUMO DAS PREVISÕES:")
    print(f"Estatísticas: min={min_demanda}, max={max_demanda}, média={media_demanda:.2f}")
    
    # Cria uma cópia para exibição com marca de data atual
    resultados_exibicao = resultados.copy()
    resultados_exibicao['data'] = resultados_exibicao['data'] + ' (atual)'
    print(resultados_exibicao[['data', 'categoria', 'preco_unitario', 'promocao', 'dia_da_semana', 'demanda_prevista']])


def main():
    """Função principal que executa o fluxo completo de análise de demanda."""
    print("\n" + "="*80)
    print("ANÁLISE DE DEMANDA COMPLETA - ShopFast")
    print("="*80)
    
    # Configura pasta de saída para os resultados
    pasta_resultados, pasta_graficos = configurar_pasta_saida()
    
    # Carrega o modelo
    model, model_columns = carregar_modelo()
    if model is None or model_columns is None:
        return
    
    # Dados para previsão
    novos_produtos = [
        {
            "categoria": "Eletrônicos",
            "preco_unitario": 450.00,
            "promocao": "Sim",
            "temperatura_media": 29.0,
            "humidade_media": 60.0,
            "dia_da_semana": "Sexta",
            "feedback_cliente": 5
        },
        {
            "categoria": "Roupas",
            "preco_unitario": 89.90,
            "promocao": "Não",
            "temperatura_media": 22.0,
            "humidade_media": 75.0,
            "dia_da_semana": "Sábado",
            "feedback_cliente": 4
        },
        {
            "categoria": "Utensílios",
            "preco_unitario": 35.50,
            "promocao": "Não",
            "temperatura_media": 28.0,
            "humidade_media": 62.0,
            "dia_da_semana": "Sexta",
            "feedback_cliente": 3
        }
    ]
    
    print("\n1. PREVISÃO PARA NOVOS PRODUTOS")
    print("-"*40)
    resultados, data_atual = prever_novos_produtos(model, model_columns, novos_produtos)
    
    # Exibe e salva os resultados das previsões
    exibir_detalhe_previsoes(resultados, data_atual)
    arquivo_saida = os.path.join(pasta_resultados, f'previsao_demanda_{data_atual.replace("-", "_")}.csv')
    resultados.to_csv(arquivo_saida, index=False)
    print(f"\nResultados salvos em: {arquivo_saida}")
    
    # Gera o gráfico de barras para as previsões
    print("\n2. GERANDO VISUALIZAÇÕES")
    print("-"*40)
    caminho_grafico = gerar_grafico_barras(resultados, data_atual, pasta_graficos)
    print(f"Gráfico de barras salvo em: {caminho_grafico}")
    
    # Processa e analisa o dataset histórico completo
    print("\n3. ANÁLISE DO DATASET HISTÓRICO")
    print("-"*40)
    caminho_dataset = 'dados/2025.1 - Vendas_semestre.txt'
    df_historico = prever_dataset_historico(model, model_columns, caminho_dataset)
    
    if df_historico is not None:
        # Salva o resultado do dataset histórico
        arquivo_historico = os.path.join(pasta_resultados, 'analise_dataset_completo.csv')
        df_historico.to_csv(arquivo_historico, index=False, sep=';')
        print(f"Dataset histórico com previsões salvo em: {arquivo_historico}")
        
        # Gera gráficos para o dataset histórico
        caminhos_graficos = gerar_grafico_historico(df_historico, pasta_graficos)
        print("Gráficos de análise histórica salvos em:")
        for caminho in caminhos_graficos:
            print(f"  - {caminho}")
    
    print("\n" + "="*80)
    print(f"ANÁLISE COMPLETA FINALIZADA! Todos os resultados estão na pasta: {pasta_resultados}")
    print("="*80)
    
    # Exibe os gráficos (isso abrirá janelas com os gráficos)
    plt.show()


if __name__ == "__main__":
    main()
