# Algoritmo de Análise de Demanda

## Visão Geral

Este algoritmo realiza a previsão de demanda utilizando machine learning com base em dados históricos de vendas e fatores contextuais, recomendando quais produtos devem ser comprados.

## Como Funciona

O algoritmo de previsão de demanda funciona em quatro etapas principais:

1. **Processamento de Dados**: Limpa e transforma dados históricos de vendas, incluindo:

   - Limpeza (remoção de valores nulos, tratamento de outliers)
   - Análise exploratória para identificar padrões e correlações
   - Engenharia de características (extração de componentes temporais, codificação categórica)

2. **Treinamento do Modelo**: Utiliza machine learning para criar modelo preditivo:

   - Divisão em conjuntos de treino e teste
   - Seleção do algoritmo (Random Forest ou Gradient Boosting)
   - Ajuste de hiperparâmetros e avaliação de desempenho

3. **Geração de Previsões**: Aplica o modelo para prever demandas futuras:
   - Preparação dos dados existentes para previsão
   - Aplicação do modelo treinado
   - Interpretação e formatação dos resultados

4. **Recomendação de Produtos**: Analisa os produtos existentes e recomenda quais comprar:
   - Comparação da demanda prevista com a média histórica
   - Classificação de produtos por prioridade de compra
   - Visualização gráfica das recomendações

## Variáveis Consideradas

- Data (para capturar tendências e sazonalidades)
- Categoria do Produto
- Preço Unitário
- Status de Promoção
- Fatores ambientais (temperatura, umidade)
- Dia da Semana
- Feedback do Cliente

## Como Executar

### Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

### Execução

```bash
# Modo padrão (treinamento e recomendação)
python main.py

# Modo de treinamento com dados específicos
python main.py --dados caminho/para/dados.csv
```

## Limitações

- Requer dados históricos suficientes para treinamento eficaz
- Necessita retrainamento periódico para manter precisão
- Sensível a mudanças bruscas no mercado

## Detalhes do Algoritmo

### 1. Processamento de Dados

O processamento de dados envolve várias etapas críticas:

- **Limpeza de Dados**: Remoção de entradas duplicadas, preenchimento de valores ausentes e correção de tipos de dados.
- **Tratamento de Outliers**: Identificação e remoção de valores extremos que podem distorcer a análise.
- **Análise Exploratória**: Geração de estatísticas descritivas e visualizações para entender a distribuição e relações nos dados.

```python
def carregar_e_processar_dados(caminho_dados, delimiter=','):
    """Carrega e processa os dados para treinamento."""
    print("\n1. Carregando e pré-processando os dados...")
    data_loader = DataLoader(file_path=caminho_dados, delimiter=delimiter)
    
    # Carregar dados com validação de colunas
    data = data_loader.load_data(required_columns=[...])
    
    # Pré-processar dados
    X_train, X_test, y_train, y_test, product_ids_test = data_loader.preprocess_data()
    
    return {...}
```

### 2. Engenharia de Características

Com base na análise exploratória, o algoritmo cria características relevantes:

- Extração de componentes temporais (mês, dia da semana)
- Codificação one-hot para variáveis categóricas
- Normalização de variáveis numéricas

### 3. Treinamento do Modelo

O algoritmo treina um modelo preditivo usando os dados preparados:

- Divisão em conjuntos de treino e teste
- Seleção do algoritmo (Random Forest por padrão)
- Ajuste de hiperparâmetros para otimizar o desempenho

```python
def treinar_modelo(dados_processados, model_type="random_forest"):
    """Treina o modelo de previsão de demanda."""
    print("\n2. Construindo e treinando o modelo...")
    model_builder = ModelBuilder(model_type=model_type, max_depth=10, random_state=42)
    
    model = model_builder.train(
        dados_processados['X_train'], 
        dados_processados['y_train'], 
        dados_processados['feature_names']
    )
    
    return {...}
```

### 4. Avaliação do Modelo

O algoritmo avalia a qualidade do modelo usando:

- Métricas de erro (MAE, RMSE)
- Coeficiente de determinação (R²)
- Análise de resíduos

### 5. Recomendação de Produtos

O algoritmo analisa os produtos existentes e recomenda quais devem ser comprados:

```python
def recomendar_produtos():
    """Recomenda produtos para compra com base nas previsões de demanda."""
    print("\n6. Analisando produtos existentes para recomendações de compra...")
    
    # Carregar modelo treinado
    model = joblib.load('modelos/modelo_demanda.pkl')
    
    # Carregar e analisar dados históricos
    dados = pd.read_csv('dados/2025.1 - Vendas_semestre.txt', sep=',')
    
    # Agrupar dados por produto
    produtos_dados = dados.groupby('produto_id').agg({...})
    
    # Obter as últimas tendências para cada produto
    tendencias = dados.sort_values('data').groupby('produto_id').tail(5)
    
    # Fazer previsão para cada produto
    previsoes = model.predict(X_pred)
    
    # Classificar produtos por prioridade
    resultados['recomendacao'] = 'Normal'
    resultados.loc[resultados['demanda_prevista'] > resultados['venda_media'] * 1.2, 'recomendacao'] = 'Alta'
    resultados.loc[resultados['demanda_prevista'] < resultados['venda_media'] * 0.8, 'recomendacao'] = 'Baixa'
    
    return resultados
```

### 6. Visualizações

O algoritmo gera visualizações para facilitar a interpretação:

- Importância das features no modelo
- Comparação entre valores reais e previstos
- Distribuição dos erros
- Recomendações de produtos para compra

## Variáveis Utilizadas

O modelo considera as seguintes variáveis para fazer previsões:

- **Data**: Para capturar tendências e sazonalidades
- **Categoria do Produto**: Tipo de produto sendo vendido
- **Preço Unitário**: Valor de venda do produto
- **Promoção**: Indicador se o produto estava em promoção
- **Temperatura Média**: Condição climática do dia
- **Umidade Média**: Nível de umidade do ambiente
- **Dia da Semana**: Para capturar padrões semanais de consumo
- **Feedback do Cliente**: Nível de satisfação do cliente

## Personalização

O algoritmo pode ser personalizado de várias maneiras:

1. **Algoritmo de Aprendizado**: Substituir o modelo atual por outro algoritmo
2. **Engenharia de Características**: Adicionar novas características ou transformar as existentes
3. **Hiperparâmetros**: Ajustar os parâmetros do modelo para melhor desempenho

## Limitações Conhecidas

- Requer dados históricos suficientes para treinamento eficaz
- Sensível a mudanças bruscas no mercado não representadas nos dados históricos
- Necessita de retrainamento periódico para manter a precisão

## Extensões Futuras

- Implementação de modelos de séries temporais (ARIMA, Prophet)
- Integração com APIs externas para dados de clima em tempo real
- Interface web para visualização interativa dos resultados

## Conclusão

O algoritmo de análise de demanda fornece uma ferramenta poderosa para prever necessidades futuras de estoque e vendas. Em vez de gerar produtos fictícios, ele analisa o histórico de produtos existentes para recomendar quais devem ser comprados, permitindo decisões mais informadas sobre gerenciamento de inventário e planejamento de negócios.
