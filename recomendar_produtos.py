def recomendar_produtos():
    """Recomenda produtos para compra com base nas previsões de demanda para produtos existentes."""
    print("\n6. Analisando produtos existentes para recomendações de compra...")
    
    # Carregar modelo e metadados
    try:
        # Verificar se modelo existe
        modelo_path = 'modelos/modelo_demanda.pkl'
        colunas_path = 'modelos/colunas_modelo.pkl'
        
        if not os.path.exists(modelo_path) or not os.path.exists(colunas_path):
            print(f"Modelo ou colunas não encontrados. Execute o treinamento primeiro.")
            return None
        
        # Carregar modelo e feature names    
        model = joblib.load(modelo_path)
        feature_names = joblib.load(colunas_path)  # Carregar os nomes das features usados no treinamento
        print(f"Modelo carregado com {len(feature_names)} features.")
        
        # Tentar carregar preprocessador, mas criar um novo se não existir
        preprocessor_path = 'modelos/preprocessor_modelo.pkl'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            print("Preprocessador carregado com sucesso.")
        else:
            print("AVISO: Preprocessador não encontrado. Usando StandardScaler padrão.")
            preprocessor = StandardScaler()
        
        # Carregar dados originais
        caminho_dados = os.path.join('dados', '2025.1 - Vendas_semestre.txt')
        dados = pd.read_csv(caminho_dados, sep=',')
        
        # Limpar nomes das colunas (remover espaços extras)
        dados.columns = dados.columns.str.strip()
        
        # Filtrar apenas linhas com datas válidas
        dados['data'] = pd.to_datetime(dados['data'], errors='coerce')
        dados = dados.dropna(subset=['data'])
        
        # Pré-processar para previsão
        # Adicionar features temporais mais detalhadas (mesmas do treinamento)
        dados['mes'] = dados['data'].dt.month
        dados['dia'] = dados['data'].dt.day
        dados['ano'] = dados['data'].dt.year
        dados['semana_do_ano'] = dados['data'].dt.isocalendar().week
        dados['dia_do_ano'] = dados['data'].dt.dayofyear
        dados['e_fim_de_semana'] = dados['dia_da_semana'].isin([5, 6, 'Sábado', 'Domingo', 'Saturday', 'Sunday']).astype(int)
        
        # Codificação cíclica para variáveis sazonais
        dados['mes_sin'] = np.sin(2 * np.pi * dados['mes']/12)
        dados['mes_cos'] = np.cos(2 * np.pi * dados['mes']/12)
        dados['dia_sin'] = np.sin(2 * np.pi * dados['dia']/31)
        dados['dia_cos'] = np.cos(2 * np.pi * dados['dia']/31)
        
        # Features de interação
        dados['temp_umidade'] = dados['temperatura_media'] * dados['humidade_media']
        
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
        
        # Preparar dados para previsão - usar os dados originais para ter acesso a todas as features
        tendencias = dados.sort_values('data').groupby('produto_id').tail(5)
        
        # Para cada produto, obter os valores mais recentes para prever a próxima demanda
        produtos_para_previsao = []
        for produto_id in produtos_dados['produto_id'].unique():
            ultimos_dados = tendencias[tendencias['produto_id'] == produto_id].iloc[-1:].copy()
            
            if not ultimos_dados.empty:
                # Adicionar features de data
                ultimos_dados['data_previsao'] = pd.to_datetime('today')
                ultimos_dados['ano'] = ultimos_dados['data_previsao'].dt.year
                ultimos_dados['mes'] = ultimos_dados['data_previsao'].dt.month
                ultimos_dados['dia'] = ultimos_dados['data_previsao'].dt.day
                ultimos_dados['semana_do_ano'] = ultimos_dados['data_previsao'].dt.isocalendar().week
                ultimos_dados['dia_do_ano'] = ultimos_dados['data_previsao'].dt.dayofyear
                
                # Garantir que todas as colunas calculadas existam
                ultimos_dados['mes_sin'] = np.sin(2 * np.pi * ultimos_dados['mes']/12)
                ultimos_dados['mes_cos'] = np.cos(2 * np.pi * ultimos_dados['mes']/12)
                ultimos_dados['dia_sin'] = np.sin(2 * np.pi * ultimos_dados['dia']/31)
                ultimos_dados['dia_cos'] = np.cos(2 * np.pi * ultimos_dados['dia']/31)
                ultimos_dados['temp_umidade'] = ultimos_dados['temperatura_media'] * ultimos_dados['humidade_media']
                
                produtos_para_previsao.append(ultimos_dados)
        
        if not produtos_para_previsao:
            print("Não foi possível preparar dados para previsão.")
            return None
            
        # Concatenar todos os produtos para previsão
        df_previsao = pd.concat(produtos_para_previsao, ignore_index=True)
        
        # Remover colunas não usadas no modelo e a quantidade vendida (target)
        df_previsao = df_previsao.drop(['quantidade_vendida', 'data', 'data_previsao'], axis=1, errors='ignore')
        
        # Fazer one-hot encoding
        df_previsao_encoded = pd.get_dummies(df_previsao, drop_first=True)
        
        # Garantir que todas as colunas usadas no treinamento estejam presentes
        print(f"Colunas nos dados de previsão antes do ajuste: {len(df_previsao_encoded.columns)}")
        for col in feature_names:
            if col not in df_previsao_encoded.columns:
                print(f"Adicionando coluna ausente: {col}")
                df_previsao_encoded[col] = 0
        
        # Garantir que apenas as colunas usadas no treinamento sejam usadas e na mesma ordem
        df_previsao_final = df_previsao_encoded[feature_names]
        print(f"Colunas nos dados de previsão após ajuste: {len(df_previsao_final.columns)}")
        
        try:
            # Fazer previsão com os dados preparados corretamente
            previsoes = model.predict(df_previsao_final)
            print("Previsão realizada com sucesso!")
        except Exception as predict_error:
            print(f"Erro ao fazer previsão: {str(predict_error)}")
            return None
        
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
