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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# Tentar importar XGBoost - tornar opcional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("AVISO: Biblioteca XGBoost não encontrada. Usando RandomForest como alternativa.")
    print("Para instalar XGBoost, execute: pip install xgboost")
    XGBOOST_AVAILABLE = False

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
    """Carrega e processa os dados para treinamento com melhorias."""
    print("\n1. Carregando e pré-processando os dados...")
    data_loader = DataLoader(file_path=caminho_dados, delimiter=delimiter)
    
    # Definir colunas obrigatórias mínimas
    colunas_necessarias = [
        'data', 'produto_id', 'categoria', 'quantidade_vendida',
        'preco_unitario', 'temperatura_media', 'humidade_media', 'dia_da_semana'
    ]
    
    # Carregar dados
    data = data_loader.load_data(required_columns=colunas_necessarias)
    
    # MELHORIA: Verificar e remover outliers
    print("   - Verificando e tratando outliers...")
    target_column = 'quantidade_vendida'
    q1 = data[target_column].quantile(0.01)
    q3 = data[target_column].quantile(0.99)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_mask = (data[target_column] < lower_bound) | (data[target_column] > upper_bound)
    print(f"     Outliers encontrados: {outliers_mask.sum()} ({outliers_mask.sum()/len(data)*100:.2f}%)")
    
    # Opção 1: Remover outliers (para conjuntos grandes)
    if len(data) > 1000:
        data_cleaned = data[~outliers_mask].copy()
        print(f"     Removendo {outliers_mask.sum()} outliers.")
    # Opção 2: Limitar valores (para conjuntos menores)
    else:
        data_cleaned = data.copy()
        data_cleaned.loc[data_cleaned[target_column] < lower_bound, target_column] = lower_bound
        data_cleaned.loc[data_cleaned[target_column] > upper_bound, target_column] = upper_bound
        print("     Limitando valores de outliers.")
    
    # MELHORIA: Engenharia de features avançada
    print("   - Aplicando engenharia de features avançada...")
    
    # Converter data para datetime
    data_cleaned['data'] = pd.to_datetime(data_cleaned['data'])
    
    # Features temporais mais detalhadas
    data_cleaned['ano'] = data_cleaned['data'].dt.year
    data_cleaned['mes'] = data_cleaned['data'].dt.month
    data_cleaned['dia'] = data_cleaned['data'].dt.day
    data_cleaned['semana_do_ano'] = data_cleaned['data'].dt.isocalendar().week
    data_cleaned['dia_do_ano'] = data_cleaned['data'].dt.dayofyear
    data_cleaned['e_fim_de_semana'] = data_cleaned['dia_da_semana'].isin([5, 6, 'Sábado', 'Domingo', 'Saturday', 'Sunday']).astype(int)
    
    # Codificação cíclica para variáveis sazonais
    data_cleaned['mes_sin'] = np.sin(2 * np.pi * data_cleaned['mes']/12)
    data_cleaned['mes_cos'] = np.cos(2 * np.pi * data_cleaned['mes']/12)
    data_cleaned['dia_sin'] = np.sin(2 * np.pi * data_cleaned['dia']/31)
    data_cleaned['dia_cos'] = np.cos(2 * np.pi * data_cleaned['dia']/31)
    
    # Features de interação
    data_cleaned['temp_umidade'] = data_cleaned['temperatura_media'] * data_cleaned['humidade_media']
    
    # Features agregadas por produto
    produto_stats = data_cleaned.groupby('produto_id')[target_column].agg(['mean', 'median', 'std']).reset_index()
    produto_stats.columns = ['produto_id', 'produto_venda_media', 'produto_venda_mediana', 'produto_venda_std']
    data_cleaned = pd.merge(data_cleaned, produto_stats, on='produto_id', how='left')
    
    # Features agregadas por categoria
    if 'categoria' in data_cleaned.columns:
        categoria_stats = data_cleaned.groupby('categoria')[target_column].agg(['mean', 'median']).reset_index()
        categoria_stats.columns = ['categoria', 'categoria_venda_media', 'categoria_venda_mediana']
        data_cleaned = pd.merge(data_cleaned, categoria_stats, on='categoria', how='left')
    
    # Verificar dados faltantes após transformações
    missing_data = data_cleaned.isnull().sum()
    if missing_data.any():
        print("     Valores faltantes após transformação:")
        print(missing_data[missing_data > 0])
        # Preencher valores faltantes (média para numéricos, moda para categóricos)
        for col in data_cleaned.columns:
            if data_cleaned[col].dtype == 'object':
                data_cleaned[col].fillna(data_cleaned[col].mode()[0], inplace=True)
            else:
                data_cleaned[col].fillna(data_cleaned[col].mean(), inplace=True)
    
    # Ordenar dados por data para divisão temporal adequada
    data_cleaned.sort_values('data', inplace=True)
    
    # MELHORIA: Divisão temporal dos dados (80/20)
    print("   - Aplicando divisão temporal dos dados...")
    split_idx = int(len(data_cleaned) * 0.8)
    train_data = data_cleaned.iloc[:split_idx].copy()
    test_data = data_cleaned.iloc[split_idx:].copy()
    
    print(f"     Tamanho do conjunto de treino: {len(train_data)}")
    print(f"     Tamanho do conjunto de teste: {len(test_data)}")
    
    # Preparar X e y
    features_to_drop = ['data', target_column]
    
    X_train = train_data.drop(features_to_drop, axis=1)
    y_train = train_data[target_column]
    
    X_test = test_data.drop(features_to_drop, axis=1)
    y_test = test_data[target_column]
    
    # Guardar IDs dos produtos no conjunto de teste
    product_ids_test = test_data['produto_id']
    
    # MELHORIA: Normalização das features numéricas
    print("   - Normalizando features numéricas...")
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    
    X_train_num = X_train[numeric_cols]
    X_test_num = X_test[numeric_cols]
    
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    # Substituir colunas numéricas por versões normalizadas
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = X_train_num_scaled
    X_test_scaled[numeric_cols] = X_test_num_scaled
    
    # Codificar variáveis categóricas usando one-hot encoding
    X_train_encoded = pd.get_dummies(X_train_scaled, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_scaled, drop_first=True)
    
    # Garantir que X_train e X_test tenham as mesmas colunas
    for col in X_train_encoded.columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    
    # Reordenar colunas para garantir alinhamento
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Obter nomes de features finais
    feature_names = X_train_encoded.columns.tolist()
    
    return {
        'data_loader': data_loader,
        'data': data_cleaned,
        'X_train': X_train_encoded,
        'X_test': X_test_encoded, 
        'y_train': y_train,
        'y_test': y_test,
        'product_ids_test': product_ids_test,
        'feature_names': feature_names,
        'target_column': target_column,
        'scaler': scaler
    }

def treinar_modelo(dados_processados, model_type="xgboost"):
    """Treina o modelo de previsão de demanda com validação cruzada temporal e otimização."""
    print("\n2. Construindo e treinando o modelo...")
    
    # Verificar se o tipo de modelo solicitado está disponível
    if model_type == "xgboost" and not XGBOOST_AVAILABLE:
        print("   - XGBoost solicitado mas não está disponível. Usando Random Forest como alternativa.")
        model_type = "random_forest"
    
    # MELHORIA: Escolher o algoritmo adequado
    if model_type == "random_forest":
        print("   - Usando Random Forest Regressor com otimização de hiperparâmetros")
        
        # Configurar validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Definir parâmetros para otimização
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Criar modelo base
        base_model = RandomForestRegressor(random_state=42)
        
        # Aplicar grid search com validação cruzada
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Treinar modelo
        print("   - Executando otimização de hiperparâmetros (pode demorar um pouco)...")
        grid_search.fit(dados_processados['X_train'], dados_processados['y_train'])
        
        # Obter melhor modelo
        model = grid_search.best_estimator_
        print(f"   - Melhores parâmetros: {grid_search.best_params_}")
        
        # Calcular importância das features
        feature_importance = pd.DataFrame({
            'feature': dados_processados['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Renomear as colunas para corresponder ao esperado pela visualização
        feature_importance.rename(columns={
            'feature': 'Feature',
            'importance': 'Importance'
        }, inplace=True)
        
    elif model_type == "xgboost" and XGBOOST_AVAILABLE:
        print("   - Usando XGBoost Regressor com otimização de hiperparâmetros")
        
        # Configurar validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Definir parâmetros para otimização (reduzidos para economizar tempo)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1]
        }
        
        # Criar modelo base
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Aplicar grid search com validação cruzada
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Treinar modelo
        print("   - Executando otimização de hiperparâmetros (pode demorar um pouco)...")
        grid_search.fit(dados_processados['X_train'], dados_processados['y_train'])
        
        # Obter melhor modelo
        model = grid_search.best_estimator_
        print(f"   - Melhores parâmetros: {grid_search.best_params_}")
        
        # Calcular importância das features
        feature_importance = pd.DataFrame({
            'feature': dados_processados['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Renomear as colunas para corresponder ao esperado pela visualização
        feature_importance.rename(columns={
            'feature': 'Feature',
            'importance': 'Importance'
        }, inplace=True)
    
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    print(f"   - Top 10 features mais importantes:")
    for i, row in feature_importance.head(10).iterrows():
        # Corrigido: usar os nomes das colunas correspondentes
        print(f"     {row['Feature']}: {row['Importance']:.4f}")
    
    return {
        'model': model,
        'model_builder': None,  # Não estamos mais usando a classe ModelBuilder
        'feature_importance': feature_importance,
        'best_params': grid_search.best_params_
    }

def avaliar_modelo(modelo, dados_processados):
    """Avalia o desempenho do modelo com métricas mais detalhadas."""
    print("\n3. Avaliando o desempenho do modelo...")
    
    # Fazer previsões no conjunto de teste
    y_pred = modelo['model'].predict(dados_processados['X_test'])
    
    # Garantir que previsões não sejam negativas
    y_pred = np.maximum(y_pred, 0)
    
    # Calcular métricas
    mae = mean_absolute_error(dados_processados['y_test'], y_pred)
    mse = mean_squared_error(dados_processados['y_test'], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(dados_processados['y_test'], y_pred)
    
    # Calcular MAPE (Mean Absolute Percentage Error)
    # Evitando divisão por zero
    mask = dados_processados['y_test'] != 0
    y_true_masked = dados_processados['y_test'][mask]
    y_pred_masked = y_pred[mask]
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    # Exibir métricas detalhadas
    print(f"   - MAE (Erro Médio Absoluto): {mae:.2f}")
    print(f"   - MSE (Erro Quadrático Médio): {mse:.2f}")
    print(f"   - RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f}")
    print(f"   - R² (Coeficiente de Determinação): {r2:.4f}")
    print(f"   - MAPE (Erro Percentual Médio Absoluto): {mape:.2f}%")
    
    # MELHORIA: Análise de erro por faixa de valores
    print("\n   - Analisando erro por faixa de valores:")
    y_test = dados_processados['y_test'].values
    
    # Criar faixas de valores
    bins = [0, 5, 10, 20, 50, 100, float('inf')]
    labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '100+']
    
    # Atribuir cada valor a uma faixa
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
    
    # Calcular métricas por faixa
    error_by_range = {}
    for label in labels:
        mask = (y_test_binned == label)
        if mask.sum() > 0:
            range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            range_mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / np.maximum(y_test[mask], 0.01))) * 100
            count = mask.sum()
            error_by_range[label] = {'mae': range_mae, 'mape': range_mape, 'count': count}
            print(f"     Faixa {label}: MAE = {range_mae:.2f}, MAPE = {range_mape:.2f}%, Amostras = {count}")
    
    # Preparar resultados de previsão
    prediction_results = pd.DataFrame({
        'produto_id': dados_processados['product_ids_test'],
        'real': y_test,
        'previsto': y_pred,
        'erro': np.abs(y_test - y_pred),
        'erro_percentual': np.abs((y_test - y_pred) / np.maximum(y_test, 0.01)) * 100
    })
    
    return {
        'evaluator': None,  # Não estamos mais usando a classe ModelEvaluator
        'metrics': metrics,
        'predictions': y_pred,
        'prediction_results': prediction_results,
        'error_by_range': error_by_range
    }

def visualizar_resultados(modelo, avaliacao, dados_processados, pastas):
    """Gera visualizações dos resultados."""
    print("\n4. Gerando visualizações...")
    visualizer = ModelVisualizer(output_dir=pastas['graficos'])
    
    # Gerar e salvar gráficos
    visualizer.plot_feature_importance(modelo['feature_importance'])
    visualizer.plot_actual_vs_predicted(dados_processados['y_test'], avaliacao['predictions'])
    
    # Renomear colunas para compatibilidade com a função de visualização
    prediction_results = avaliacao['prediction_results'].copy()
    prediction_results.rename(columns={
        'real': 'demanda_real',
        'previsto': 'demanda_prevista'
    }, inplace=True)
    
    # Usar o DataFrame com colunas renomeadas
    visualizer.plot_prediction_results(prediction_results)
    
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
    
    # Salvar o scaler (preprocessador) que criamos durante o processamento de dados
    if 'scaler' in dados_processados:
        joblib.dump(dados_processados['scaler'], preprocessor_path)
        print(f"Preprocessador salvo em: {preprocessor_path}")
    else:
        print("AVISO: Nenhum preprocessador disponível para salvar")
    
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
        # Verificar se modelo existe
        modelo_path = 'modelos/modelo_demanda.pkl'
        if not os.path.exists(modelo_path):
            print(f"Modelo não encontrado em {modelo_path}. Execute o treinamento primeiro.")
            return None
            
        model = joblib.load(modelo_path)
        
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
        # Se o preprocessador for None, criar um novo
        if preprocessor is None:
            print("AVISO: Preprocessador inválido. Criando um novo StandardScaler.")
            preprocessor = StandardScaler()
            # Ajustar o preprocessador aos dados atuais
            numeric_cols = df_previsao.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                preprocessor.fit(df_previsao[numeric_cols])
        
        try:
            # Tentar transformar os dados
            X_pred = preprocessor.transform(df_previsao)
        except Exception as transform_error:
            print(f"Erro ao transformar dados: {str(transform_error)}")
            print("Tentando abordagem alternativa com dados brutos...")
            # Fallback: usar os dados brutos sem transformação
            X_pred = df_previsao.values
        
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
        
        # 2. Treinar o modelo (usando XGBoost como padrão)
        modelo_treinado = treinar_modelo(dados_processados, model_type="xgboost")
        
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
            # Tentar read as primeiras linhas para diagnóstico
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