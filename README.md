# Algoritmo de Análise de Demanda - ShopFast

## Visão Geral

Este algoritmo realiza a previsão de demanda utilizando machine learning com base em dados históricos de vendas e fatores contextuais, recomendando quais produtos devem ser comprados para otimizar o estoque.

## Estrutura do Projeto

```
AI-decision-tree-predictive-model/
├── src/
│   └── shopfast/
│       ├── data_loader.py         # Carregamento e limpeza de dados
│       ├── data_preprocessor.py   # Preparação e transformação de dados
│       ├── model_builder.py       # Construção de diferentes tipos de modelos
│       ├── model_trainer.py       # Gerenciamento do processo de treinamento
│       ├── demand_model.py        # Implementação do modelo de previsão
│       ├── model_evaluator.py     # Avaliação de desempenho do modelo
│       ├── predictor.py           # Classe para realizar previsões
│       └── visualization.py       # Visualização dos resultados
├── main.py                        # Script principal de execução
├── fazer_previsao.py              # Script para previsão com novos dados
├── prever_tudo.py                 # Script para previsão em lote
├── recomendar_produtos.py         # Script para gerar recomendações de produtos
├── modelos/                       # Diretório para armazenar modelos treinados
├── dados/                         # Diretório para arquivos de dados
└── resultados/                    # Diretório para armazenar resultados e gráficos
```

## Como Funciona

O algoritmo de previsão de demanda funciona em cinco etapas principais:

1. **Processamento de Dados**: Limpa e transforma dados históricos de vendas, incluindo:
   - Limpeza (remoção de valores nulos, tratamento de outliers)
   - Análise exploratória para identificar padrões e correlações
   - Engenharia de características (extração de componentes temporais, codificação categórica)
   - Tratamento de dados temporais com codificação cíclica (seno/cosseno)

2. **Treinamento do Modelo**: Utiliza machine learning para criar modelo preditivo:
   - Divisão em conjuntos de treino e teste
   - Seleção do algoritmo (Decision Tree, Random Forest, XGBoost ou Gradient Boosting)
   - Validação cruzada temporal para séries temporais
   - Ajuste de hiperparâmetros e avaliação de desempenho

3. **Avaliação do Modelo**: Analisa o desempenho do modelo com métricas relevantes:
   - R² (Coeficiente de determinação)
   - MAE (Erro médio absoluto)
   - RMSE (Raiz do erro quadrático médio)
   - MAPE (Erro percentual absoluto médio)
   - Análise de erro por faixa de valores

4. **Geração de Previsões**: Aplica o modelo para prever demandas futuras:
   - Preparação dos dados existentes para previsão
   - Aplicação do modelo treinado
   - Interpretação e formatação dos resultados
   - Visualizações para análise de resultados

5. **Recomendação de Produtos**: Analisa os produtos existentes e recomenda quais comprar:
   - Comparação da demanda prevista com a média histórica
   - Classificação de produtos por prioridade de compra baseada na relação demanda/preço
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

## Variáveis Consideradas

O modelo considera as seguintes variáveis para fazer previsões:

- **Data** (componentes temporais: mês, dia, semana do ano, etc.)
- **Produto ID**: Identificador único do produto
- **Categoria do Produto**: Tipo de produto sendo vendido
- **Preço Unitário**: Valor de venda do produto
- **Promoção**: Indicador se o produto estava em promoção
- **Temperatura Média**: Condição climática do dia
- **Umidade Média**: Nível de umidade do ambiente
- **Dia da Semana**: Para capturar padrões semanais de consumo
- **Feedback do Cliente**: Nível de satisfação do cliente

## Instalação e Uso

### Instalação

```bash
# Instalar dependências
pip install pandas numpy scikit-learn matplotlib seaborn joblib
# Opcional para melhor desempenho
pip install xgboost
```

### Execução

```bash
# Modo completo (treinamento, avaliação e recomendação)
python main.py

# Fazer previsões para novos produtos
python fazer_previsao.py

# Prever demanda para todo o dataset
python prever_tudo.py

# Gerar recomendações de produtos para compra
python recomendar_produtos.py
```

## Detalhes do Algoritmo

### Engenharia de Características

O algoritmo realiza transformações sofisticadas nos dados:

- Extração de features temporais (mês, dia, semana do ano, dia do ano)
- Codificação cíclica para variáveis sazonais (seno/cosseno)
- One-hot encoding para variáveis categóricas
- Normalização de variáveis numéricas
- Detecção e tratamento de outliers
- Identificação de fins de semana e feriados

### Modelos Suportados

O sistema suporta múltiplos algoritmos de aprendizado de máquina:

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost (quando disponível)

### Visualizações

O sistema gera visualizações para facilitar a interpretação:

- Importância das features
- Comparação entre valores reais e previstos
- Distribuição dos erros de previsão
- Gráficos de recomendações de produtos
- Análise de erro por faixa de valores

### Estrutura do Algoritmo de Recomendação

O algoritmo de recomendação de produtos funciona da seguinte forma:

```python
# Carregar modelo treinado
model = joblib.load('modelos/modelo_demanda.pkl')

# Processar dados históricos para análise
dados = pd.read_csv('dados/2025.1 - Vendas_semestre.txt', sep=',')

# Adicionar features temporais para melhorar a precisão
dados['mes'] = dados['data'].dt.month
dados['dia'] = dados['data'].dt.day
dados['semana_do_ano'] = dados['data'].dt.isocalendar().week
dados['dia_do_ano'] = dados['data'].dt.dayofyear
dados['e_fim_de_semana'] = dados['dia_da_semana'].isin([5, 6, 'Sábado', 'Domingo']).astype(int)

# Calcular métricas por produto
produtos_stats = dados.groupby('produto_id').agg({
    'quantidade_vendida': ['mean', 'median', 'std', 'sum'],
    'preco_unitario': 'mean'
})

# Fazer previsão para todos os produtos
previsoes = model.predict(X_pred)

# Calcular índice de prioridade (demanda prevista / preço)
resultados['prioridade_compra'] = resultados['demanda_prevista'] / resultados['preco_unitario']

# Classificar produtos para recomendação
resultados['recomendacao'] = 'Normal'
resultados.loc[resultados['demanda_prevista'] > resultados['venda_media'] * 1.2, 'recomendacao'] = 'Alta'
resultados.loc[resultados['demanda_prevista'] < resultados['venda_media'] * 0.8, 'recomendacao'] = 'Baixa'

return resultados
```

## Personalização

O algoritmo pode ser personalizado de várias maneiras:

1. **Algoritmo de Aprendizado**: Escolher entre diferentes modelos via parâmetro `model_type`
2. **Engenharia de Características**: Adicionar ou modificar features no preprocessador
3. **Hiperparâmetros**: Ajustar parâmetros como `max_depth` ou `n_estimators`
4. **Estratégia de Recomendação**: Modificar os critérios de priorização de produtos
5. **Visualizações**: Personalizar os gráficos gerados pelo sistema

## Limitações Conhecidas

- Depende da qualidade e quantidade dos dados históricos
- Requer retreinamento periódico para capturar novas tendências
- Sensível a mudanças bruscas no mercado não refletidas nos dados
- Requer tratamento adequado para novas categorias de produtos

## Extensões Futuras

- Implementação de técnicas de deep learning para séries temporais (LSTM)
- Integração com fontes externas de dados (clima, economia, eventos sazonais)
- Interface de usuário para exploração interativa dos resultados
- Previsão automática e agendada para recomendações contínuas
- Implementação de ensemble de modelos para maior precisão

## Conclusão

O algoritmo de análise de demanda ShopFast fornece uma ferramenta poderosa para prever necessidades futuras de estoque e vendas. Ele analisa o histórico de produtos existentes para recomendar quais devem ser comprados, permitindo decisões mais informadas sobre gerenciamento de inventário e planejamento de negócios. A arquitetura modular e extensível do sistema permite adaptação a diferentes contextos e conjuntos de dados.
