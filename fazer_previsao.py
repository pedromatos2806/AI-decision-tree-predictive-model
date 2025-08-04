# fazer_previsao.py

import pandas as pd
import joblib
import datetime  # Adicionado para obter a data atual

def carregar_modelo_e_prever(dados_para_prever, usar_data_atual=False):
    """
    Carrega o modelo treinado e as colunas, e faz a previsão para novos dados.
    
    Args:
        dados_para_prever: Lista de dicionários com os dados para previsão
        usar_data_atual: Se True, substitui a data em todos os registros pela data atual
    """
    try:
        # Carrega o modelo e as colunas dos arquivos salvos
        model = joblib.load('modelo_demanda.pkl')
        model_columns = joblib.load('colunas_modelo.pkl')
        print("Modelo 'modelo_demanda.pkl' carregado com sucesso.")
    except FileNotFoundError:
        print("ERRO: Arquivos de modelo não encontrados. Rode o script 'main.py' primeiro para treinar e salvar o modelo.")
        return

    # Converte os novos dados para um DataFrame do pandas
    df_novos_dados = pd.DataFrame(dados_para_prever)
    
    # Se solicitado, substitui a data pela data atual
    if usar_data_atual:
        data_atual = datetime.datetime.now().strftime('%Y-%m-%d')
        df_novos_dados['data'] = data_atual
        print(f"Usando data atual para previsão: {data_atual}")
    
    # Adiciona a coluna 'mes' a partir da data
    df_novos_dados['mes'] = pd.to_datetime(df_novos_dados['data']).dt.month
    
    # Aplica o one-hot encoding
    df_encoded = pd.get_dummies(df_novos_dados)
    
    # Reorganiza as colunas para que fiquem na mesma ordem do modelo treinado,
    # preenchendo com False as colunas que não apareceram nos novos dados.
    df_final = df_encoded.reindex(columns=model_columns, fill_value=False)
    
    # Faz a previsão
    previsoes = model.predict(df_final)
    
    # Adiciona as previsões ao DataFrame original para fácil visualização
    df_novos_dados['demanda_prevista'] = previsoes.astype(int)
    
    return df_novos_dados

if __name__ == '__main__':
    # --- AQUI VOCÊ COLOCA OS DADOS QUE QUER PREVER ---
    # Crie uma lista de dicionários, onde cada dicionário é um produto/cenário.
    
    novos_produtos = [
        {
            "data": "2025-08-08",
            "categoria": "Eletrônicos",
            "preco_unitario": 450.00,
            "promocao": "Sim",
            "temperatura_media": 29.0,
            "humidade_media": 60.0,
            "dia_da_semana": "Sexta",
            "feedback_cliente": 5
        },
        {
            "data": "2025-08-09",
            "categoria": "Roupas",
            "preco_unitario": 89.90,
            "promocao": "Não",
            "temperatura_media": 22.0,
            "humidade_media": 75.0,
            "dia_da_semana": "Sábado",
            "feedback_cliente": 4
        },
        {
            "data": "2025-08-08",
            "categoria": "Utensílios",
            "preco_unitario": 35.50,
            "promocao": "Não",
            "temperatura_media": 28.0,
            "humidade_media": 62.0,
            "dia_da_semana": "Sexta",
            "feedback_cliente": 3
        }
    ]

    # Usa a data atual para todas as as previsões
    data_atual = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"\n===== PREVISÕES DE DEMANDA PARA A DATA ATUAL: {data_atual} =====")
    
    # Aplica a data atual em todas as entradas
    for produto in novos_produtos:
        produto['data'] = data_atual
    
    # Chama a função para obter as previsões com a data atual
    resultados = carregar_modelo_e_prever(novos_produtos)
    
    if resultados is not None:
        # Configura o pandas para exibir todas as colunas e sem truncamento
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        # Formata o DataFrame para exibição mais clara
        print("\n" + "="*80)
        print("DETALHES DAS PREVISÕES:")
        print("="*80)
        
        for i, row in resultados.iterrows():
            print(f"\nItem #{i+1}:")
            print(f"  Data: {row['data']} {'(data atual)' if row['data'] == data_atual else ''}")
            print(f"  Categoria: {row['categoria']}")
            print(f"  Preço unitário: R$ {row['preco_unitario']:.2f}")
            print(f"  Promoção: {row['promocao']}")
            print(f"  Temperatura média: {row['temperatura_media']}°C")
            print(f"  Umidade média: {row['humidade_media']}%")
            print(f"  Dia da semana: {row['dia_da_semana']}")
            print(f"  Feedback cliente: {row['feedback_cliente']}")
            print(f"  DEMANDA PREVISTA: {row['demanda_prevista']} unidades")
            print("-"*50)
        
        # Mostra tabela resumida
        print("\nRESUMO DAS PREVISÕES:")
        # Adiciona coluna que indica se é a data atual
        resultados_exibicao = resultados.copy()
        resultados_exibicao['data'] = resultados_exibicao['data'] + ' (atual)'
        
        # Calcula estatísticas para mostrar no resumo
        min_demanda = resultados['demanda_prevista'].min()
        max_demanda = resultados['demanda_prevista'].max()
        media_demanda = resultados['demanda_prevista'].mean()
        
        print(f"Estatísticas das previsões: min={min_demanda}, max={max_demanda}, média={media_demanda:.2f}")
        print(resultados_exibicao[['data', 'categoria', 'preco_unitario', 'promocao', 'dia_da_semana', 'demanda_prevista']])
        
        # Salva os resultados em um arquivo CSV
        arquivo_saida = f'previsao_demanda_{data_atual.replace("-", "_")}.csv'
        resultados.to_csv(arquivo_saida, index=False)
        print(f"\nResultados salvos com sucesso no arquivo: '{arquivo_saida}'")
        
        # Gera gráfico de visualização dos resultados
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            print("\nGerando gráfico de visualização das previsões...")
            
            # Configurações do gráfico
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
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=cor, label=cat) for cat, cor in cores.items()]
            plt.legend(handles=legend_elements, title='Categorias', loc='upper right')
            
            # Ajusta o layout e exibe o grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Cria uma pasta para salvar os gráficos se não existir
            import os
            pasta_graficos = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
            if not os.path.exists(pasta_graficos):
                os.makedirs(pasta_graficos)
                print(f"Criada pasta para gráficos: {pasta_graficos}")
            
            # Salva o gráfico como imagem na pasta específica
            nome_arquivo = f'grafico_demanda_{data_atual.replace("-", "_")}.png'
            caminho_completo = os.path.join(pasta_graficos, nome_arquivo)
            plt.savefig(caminho_completo, dpi=300)  # Aumenta a resolução para 300 DPI
            
            print(f"\nGráfico salvo em:")
            print(f"  {caminho_completo}")
            
            # Exibe o gráfico (isso abrirá uma janela com o gráfico)
            plt.show()
            
        except ImportError:
            print("\nNão foi possível gerar o gráfico. Certifique-se de que a biblioteca matplotlib está instalada.")
            print("Você pode instalá-la com: pip install matplotlib")
        
        print("="*80)