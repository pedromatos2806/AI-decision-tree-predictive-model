# Guia de Explicação do Algoritmo de Previsão de Demanda

## Introdução

Prezado professor,

Este guia explica passo a passo o funcionamento do algoritmo de previsão de demanda desenvolvido para o sistema ShopFast. O algoritmo utiliza técnicas de aprendizado de máquina para prever a quantidade de produtos que serão vendidos e recomendar quais produtos devem ser comprados com base em diversos fatores contextuais.

## Estrutura do Projeto

O sistema está organizado em módulos para facilitar a manutenção e compreensão:

```
IA/
├── src/
│   └── shopfast/
│       ├── data_loader.py       # Carregamento e limpeza de dados
│       ├── model_builder.py     # Construção e treinamento do modelo
│       ├── model_evaluator.py   # Avaliação de desempenho do modelo
│       └── visualization.py     # Visualização dos resultados
├── main.py                      # Script principal de execução
├── dados/                       # Diretório para arquivos de dados
│   └── 2025.1 - Vendas_semestre.txt  # Dados históricos de vendas
├── resultados/                  # Diretório para resultados e visualizações
│   └── [data]/
│       ├── previsoes_demanda.csv   # Resultados das previsões
│       └── graficos/               # Visualizações geradas
└── modelos/                     # Modelos salvos para uso futuro
```

## Fluxo do Algoritmo

O algoritmo segue um fluxo em seis etapas principais:

1. **Carregamento dos Dados**
2. **Pré-processamento**
3. **Treinamento do Modelo**
4. **Avaliação do Desempenho**
5. **Geração de Visualizações**
6. **Recomendação de Produtos**

Vamos explorar cada uma destas etapas em detalhes.

## 1. Carregamento dos Dados

### Módulo: `data_loader.py`

O carregamento de dados é gerenciado pela classe `DataLoader`, que:

- Lê arquivos CSV ou TXT com dados históricos de vendas
- Identifica e corrige problemas de formato (delimitadores, codificação)
- Remove linhas com datas inválidas ou incompletas
- Valida a presença das colunas necessárias

**Código-chave:**

```python
def load_data(self, required_columns=None, encoding='utf-8'):
    """Carrega os dados do arquivo."""
    # Tentar diferentes codificações se necessário
    encodings = [encoding, 'latin1', 'ISO-8859-1', 'cp1252']
    
    for enc in encodings:
        try:
            # Carregar o arquivo com tratamento de linhas problemáticas
            self.data = pd.read_csv(
                self.file_path, 
                sep=self.delimiter,
                encoding=enc,
                on_bad_lines='skip'
            )
            print(f"Dados carregados com sucesso usando codificação {enc}.")
            break
        except UnicodeDecodeError:
            print(f"Erro de codificação com {enc}, tentando próxima...")
            
    # Remover linhas com datas inválidas
    if 'data' in self.data.columns:
        # Filtrar linhas com formato de data correto
        import re
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        mask = self.data['data'].astype(str).apply(lambda x: bool(date_pattern.match(x)))
        
        self.data = self.data[mask]
```

## 2. Pré-processamento dos Dados

### Módulo: `data_loader.py`

O pré-processamento é realizado dentro da classe `DataLoader`, método `preprocess_data`, que:

- Converte a coluna de data para formato datetime
- Extrai características temporais (mês, dia)
- Converte variáveis categóricas em formato numérico
- Divide os dados em conjuntos de treinamento e teste
- Padroniza as variáveis numéricas

**Código-chave:**

```python
def preprocess_data(self, target_column='quantidade_vendida', test_size=0.2, random_state=42):
    """Prepara os dados para treinamento e teste."""
    # Converter data para timestamp
    self.data['data'] = pd.to_datetime(self.data['data'], errors='coerce')
    
    # Extrair características de data
    self.data['mes'] = self.data['data'].dt.month
    self.data['dia'] = self.data['data'].dt.day
    
    # Converter 'promocao' para binário
    self.data['promocao'] = self.data['promocao'].apply(
        lambda x: 1 if x in ['Sim', 's', 'yes', 'y', '1', 'true'] else 0
    )
    
    # Divisão dos dados
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
```

## 3. Treinamento do Modelo

### Módulo: `model_builder.py`

O treinamento do modelo é gerenciado pela classe `ModelBuilder`, que:

- Suporta diferentes tipos de modelos (Decision Tree, Random Forest, Gradient Boosting)
- Treina o modelo com os dados processados
- Calcula e armazena a importância das features

**Código-chave:**

```python
def train(self, X_train, y_train, feature_names=None):
    """Treina o modelo de previsão de demanda."""
    print(f"Treinando o modelo de {type(self.model).__name__}...")
    
    self.model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
    
    # Calcular importância das features
    if hasattr(self.model, 'feature_importances_'):
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
```

### Escolha do Modelo: Random Forest

Por que utilizamos Random Forest como modelo padrão:

1. **Alta precisão**: Combina múltiplas árvores de decisão para melhorar a acurácia
2. **Robustez contra overfitting**: A combinação de múltiplos modelos reduz o risco de sobreajuste
3. **Capacidade de lidar com dados não-lineares**: Captura relações complexas entre variáveis
4. **Avaliação automática da importância das features**: Identifica quais variáveis mais impactam a previsão

## 4. Avaliação do Modelo

### Módulo: `model_evaluator.py`

A avaliação do modelo é feita através da classe `ModelEvaluator`, que:

- Calcula múltiplas métricas de desempenho (R², MAE, RMSE, MAPE)
- Compara resultados reais com previstos
- Gera um DataFrame com os resultados das previsões para cada produto

**Código-chave:**

```python
def evaluate(self, model, X_test, y_test, product_ids=None):
    """Avalia o modelo e retorna métricas de desempenho."""
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(1, y_test))) * 100
    
    # Criar DataFrame com resultados por produto
    prediction_df = pd.DataFrame({
        'produto_id': product_ids,
        'demanda_real': y_test,
        'demanda_prevista': np.round(y_pred, 0).astype(int)
    })
```

## 5. Visualização dos Resultados

### Módulo: `visualization.py`

A classe `ModelVisualizer` gera visualizações para interpretar os resultados:

- Gráfico de importância das features
- Gráfico de dispersão de valores reais vs. previstos
- Gráfico de barras comparando demanda real e prevista
- Gráfico de distribuição dos erros

## 6. Recomendação de Produtos

### Módulo: `main.py` (função `recomendar_produtos`)

Esta é uma nova funcionalidade que analisa os produtos existentes e recomenda quais devem ser comprados:

- Carrega o modelo treinado
- Analisa o histórico de vendas de cada produto
- Calcula estatísticas por produto (média, total, etc.)
- Obtém as tendências mais recentes
- Faz previsões para cada produto
- Classifica os produtos em prioridades de compra:
  - **Alta**: Quando a demanda prevista é 20% maior que a média histórica
  - **Normal**: Quando a demanda prevista está próxima da média histórica
  - **Baixa**: Quando a demanda prevista é 20% menor que a média histórica

**Código-chave:**

```python
def recomendar_produtos():
    """Recomenda produtos para compra com base nas previsões de demanda."""
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
    
    # Classificar produtos para recomendação
    resultados['recomendacao'] = 'Normal'
    resultados.loc[resultados['demanda_prevista'] > resultados['venda_media'] * 1.2, 'recomendacao'] = 'Alta'
    resultados.loc[resultados['demanda_prevista'] < resultados['venda_media'] * 0.8, 'recomendacao'] = 'Baixa'
```

O gráfico de recomendação gerado mostra:
- Os 15 produtos com maior demanda prevista
- A média histórica de vendas para comparação
- Cores diferentes para os níveis de prioridade (Alta, Normal, Baixa)

## Variáveis Utilizadas no Modelo

O modelo considera as seguintes variáveis para prever a demanda:

1. **Preço unitário**: Quanto custa o produto
2. **Temperatura média**: Temperatura do dia
3. **Umidade média**: Nível de umidade do ambiente
4. **Feedback do cliente**: Avaliação do cliente sobre o produto
5. **Mês e dia**: Captura sazonalidade
6. **Categoria do produto**: Tipo de produto (Eletrônicos, Roupas, etc.)
7. **Promoção**: Se o produto estava em promoção ou não
8. **Dia da semana**: Captura padrões semanais de consumo

## Fluxo de Execução

### Execução completa (main.py)

1. Configura pastas para armazenar resultados
2. Carrega e processa os dados históricos de vendas
3. Treina o modelo de previsão (Random Forest)
4. Avalia o desempenho do modelo
5. Gera visualizações dos resultados
6. Salva o modelo treinado
7. Analisa produtos existentes e faz recomendações de compra
8. Gera visualizações das recomendações

## Demonstração Prática

Para demonstrar o funcionamento do algoritmo, execute:

```
python main.py
```

Este comando realizará todas as etapas do algoritmo e gerará:
1. Arquivos de resultados na pasta 'resultados/[data_atual]/'
2. Gráficos na pasta 'resultados/[data_atual]/graficos/'
3. Modelo salvo na pasta 'modelos/'

## Resultados e Visualizações

O sistema gera as seguintes visualizações:

1. **Importância das features**: Mostra quais variáveis mais influenciam a previsão
2. **Valores reais vs. previstos**: Compara os valores reais e previstos
3. **Distribuição dos erros**: Analisa os erros de previsão
4. **Recomendação de produtos**: Mostra os produtos recomendados para compra

## Pontos Fortes do Algoritmo

1. **Robustez**: Trata problemas de formatação e dados incompletos
2. **Modularidade**: Código organizado em componentes independentes
3. **Flexibilidade**: Suporta diferentes tipos de modelos
4. **Orientação a negócios**: Além de prever, recomenda ações concretas (quais produtos comprar)
5. **Visualizações claras**: Facilita a interpretação dos resultados

## Limitações e Melhorias Futuras

1. **Exploração de técnicas de séries temporais**: Para capturar melhor padrões temporais
2. **Segmentação de produtos**: Tratar diferentes categorias com modelos específicos
3. **Otimização de hiperparâmetros**: Implementar busca em grade para melhorar a precisão
4. **Análise de fatores externos**: Incorporar eventos sazonais, feriados e tendências de mercado

## Conclusão

O algoritmo de previsão de demanda implementado vai além da simples previsão de quantidades: ele analisa o histórico de vendas para recomendar quais produtos devem ser comprados, classificando-os por prioridade. Esta abordagem orientada a decisões facilita o planejamento de estoque e compras, permitindo que o negócio otimize seus recursos.
