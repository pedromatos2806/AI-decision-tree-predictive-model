# Guia de Explicação do Algoritmo de Previsão de Demanda

## Introdução

Prezado professor,

Este guia explica passo a passo o funcionamento do algoritmo de previsão de demanda desenvolvido para o sistema ShopFast. O algoritmo utiliza técnicas de aprendizado de máquina para prever a quantidade de produtos que serão vendidos com base em diversos fatores contextuais.

## Estrutura do Projeto

O sistema está organizado em módulos para facilitar a manutenção e compreensão:

```
IA/
├── src/
│   └── shopfast/
│       ├── data_loader.py       # Carregamento e limpeza de dados
│       ├── data_preprocessor.py # Pré-processamento e engenharia de features
│       ├── demand_model.py      # Modelo de predição de demanda
│       └── predictor.py         # Interface para novas previsões
├── main.py                      # Script principal para treinamento
├── fazer_previsao.py            # Interface para previsões pontuais
├── prever_tudo.py               # Previsão para dataset completo
└── analise_demanda_completa.py  # Análise completa de demanda
```

## Fluxo do Algoritmo

O algoritmo segue um fluxo em cinco etapas principais:

1. **Carregamento dos Dados**
2. **Pré-processamento**
3. **Treinamento do Modelo**
4. **Avaliação do Desempenho**
5. **Realização de Previsões**

Vamos explorar cada uma destas etapas em detalhes.

## 1. Carregamento dos Dados

### Módulo: `data_loader.py`

O carregamento de dados é gerenciado pela classe `DataLoader`, que:

- Lê arquivos CSV ou TXT com dados históricos de vendas
- Realiza limpeza básica (remoção de valores nulos, correção de tipos)
- Prepara os dados para o pré-processamento

**Código-chave:**

```python
class DataLoader:
    def load_and_clean_data(self):
        # Carrega o arquivo de dados
        df = pd.read_csv(self.file_path, delimiter=self.delimiter)

        # Limpa nomes das colunas
        df.columns = df.columns.str.strip()

        # Remove linhas com valores ausentes
        df.dropna(inplace=True)

        return df
```

## 2. Pré-processamento dos Dados

### Módulo: `data_preprocessor.py`

O pré-processamento é realizado pela classe `DataPreprocessor`, que:

- Extrai características temporais (como mês) a partir da data
- Aplica codificação one-hot para variáveis categóricas
- Divide os dados em conjuntos de treinamento e teste

**Código-chave:**

```python
class DataPreprocessor:
    def preprocess(self, feature_cols, target_col):
        # Extrai o mês da data
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['mes'] = self.df['data'].dt.month

        # Seleciona as features e o alvo
        X = self.df[feature_cols]
        y = self.df[target_col]

        # Aplica one-hot encoding
        X_encoded = pd.get_dummies(X, columns=['categoria', 'promocao', 'dia_da_semana'],
                                  drop_first=True)

        # Salva a ordem das colunas para uso posterior
        self.encoder_columns = X_encoded.columns

        return X_encoded, y
```

## 3. Treinamento do Modelo

### Módulo: `demand_model.py`

O treinamento do modelo é gerenciado pela classe `DemandPredictionModel`, que:

- Inicializa um modelo de Árvore de Decisão para regressão
- Treina o modelo com os dados de treinamento
- Ajusta hiperparâmetros como profundidade máxima da árvore

**Código-chave:**

```python
class DemandPredictionModel:
    def __init__(self, max_depth=10, random_state=42):
        # Inicializa o modelo de Árvore de Decisão
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    def train(self, X_train, y_train):
        # Treina o modelo com os dados
        self.model.fit(X_train, y_train)
```

### Escolha do Modelo: Árvore de Decisão

Por que escolhemos uma Árvore de Decisão para este problema?

1. **Interpretabilidade**: É possível visualizar as regras de decisão
2. **Capacidade de lidar com relações não-lineares** entre variáveis
3. **Baixa sensibilidade a outliers** nos dados
4. **Capacidade de capturar interações** entre diferentes variáveis

## 4. Avaliação do Modelo

### Módulo: `demand_model.py`

A avaliação do modelo é feita através de:

- Coeficiente de Determinação (R²): Mede quanto da variação da demanda o modelo consegue explicar
- Erro Médio Absoluto (MAE): Mede o erro médio em unidades de produtos

**Código-chave:**

```python
def evaluate(self, X_test, y_test):
    # Faz previsões no conjunto de teste
    y_pred = self.model.predict(X_test)

    # Calcula métricas de avaliação
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Exibe resultados
    print(f"Coeficiente de Determinação (R²): {r2:.2f}")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f} unidades")
```

## 5. Realização de Previsões

### Módulo: `predictor.py`

A classe `Predictor` é responsável por:

- Receber novos dados para previsão
- Aplicar as mesmas transformações usadas no treinamento
- Fazer previsões usando o modelo treinado

**Código-chave:**

```python
class Predictor:
    def predict_demand(self, new_sale_data):
        # Cria um DataFrame a partir dos novos dados
        new_sale_df = pd.DataFrame(new_sale_data, index=[0])

        # Extrai o mês da data
        if 'mes' not in new_sale_df.columns and 'data' in new_sale_df.columns:
            new_sale_df['mes'] = pd.to_datetime(new_sale_df['data']).dt.month

        # Aplica one-hot encoding nas colunas categóricas
        if 'categoria' in new_sale_df.columns:
            new_sale_df = pd.get_dummies(new_sale_df, columns=['categoria'])

        # Garante que as colunas correspondam ao modelo treinado
        for col in self.columns:
            if col not in new_sale_df.columns:
                new_sale_df[col] = 0

        # Faz a previsão
        prediction = self.model.predict(new_sale_df)
        return int(max(0, prediction[0]))
```

## Variáveis Utilizadas no Modelo

O modelo considera as seguintes variáveis para prever a demanda:

1. **Preço unitário**: Quanto custa o produto
2. **Temperatura média**: Temperatura do dia (influencia vendas de certos produtos)
3. **Umidade média**: Nível de umidade do ambiente
4. **Feedback do cliente**: Avaliação do cliente sobre o produto
5. **Mês**: Captura sazonalidade anual
6. **Categoria do produto**: Tipo de produto (Eletrônicos, Roupas, etc.)
7. **Promoção**: Se o produto estava em promoção ou não
8. **Dia da semana**: Captura padrões semanais de consumo

## Fluxo de Execução

### Treinamento (main.py)

1. Carrega os dados históricos de vendas
2. Realiza o pré-processamento
3. Treina o modelo de Árvore de Decisão
4. Avalia o desempenho do modelo
5. Salva o modelo treinado para uso futuro

### Previsão (fazer_previsao.py)

1. Carrega o modelo treinado
2. Recebe novos dados para previsão
3. Aplica as mesmas transformações do treinamento
4. Realiza a previsão
5. Retorna a quantidade prevista de vendas

## Demonstração Prática

Para demonstrar o funcionamento do algoritmo, podemos:

1. **Mostrar o treinamento do modelo**:

   ```
   python main.py
   ```

2. **Fazer previsões para novos produtos**:

   ```
   python fazer_previsao.py
   ```

3. **Analisar o dataset completo**:
   ```
   python analise_demanda_completa.py
   ```

## Resultados e Visualizações

O sistema gera visualizações para facilitar a interpretação dos resultados:

1. **Gráficos de barras** mostrando a demanda prevista por categoria
2. **Gráficos de linhas** comparando a demanda real vs. prevista ao longo do tempo
3. **Tabelas CSV** com os resultados detalhados das previsões

## Pontos Fortes do Algoritmo

1. **Modularidade**: Código organizado em componentes reutilizáveis
2. **Tratamento robusto de dados**: Lida com valores ausentes e transformações
3. **Interpretabilidade**: Usa um modelo que permite entender as decisões
4. **Flexibilidade**: Pode ser estendido para incluir novas variáveis

## Limitações e Melhorias Futuras

1. **Exploração de outros algoritmos**: Testar Random Forest ou Gradient Boosting
2. **Otimização de hiperparâmetros**: Usar validação cruzada para melhorar o ajuste
3. **Inclusão de mais variáveis**: Como tendências de mercado ou fatores econômicos
4. **Implementação de técnicas de séries temporais**: Para capturar melhor padrões temporais

## Conclusão

O algoritmo de previsão de demanda implementado consegue capturar relações complexas entre diversas variáveis para prever com precisão razoável a quantidade de produtos que serão vendidos. A abordagem modular e flexível permite que o sistema seja facilmente adaptado e melhorado conforme necessário.
