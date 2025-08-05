# Algoritmo de Análise de Demanda - ShopFast

## Visão Geral

Este algoritmo realiza a previsão de demanda utilizando machine learning com base em dados históricos de vendas e fatores contextuais, recomendando quais produtos devem ser comprados para otimizar o estoque.

## Estrutura do Projeto

```
IA/
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
└── dados/                         # Diretório para arquivos de dados
```

## Como Funciona

O algoritmo de previsão de demanda funciona em cinco etapas principais:

1. **Processamento de Dados**: Limpa e transforma dados históricos de vendas, incluindo:
   - Limpeza (remoção de valores nulos, tratamento de outliers)
   - Análise exploratória para identificar padrões e correlações
   - Engenharia de características (extração de componentes temporais, codificação categórica)

2. **Treinamento do Modelo**: Utiliza machine learning para criar modelo preditivo:
   - Divisão em conjuntos de treino e teste
   - Seleção do algoritmo (Random Forest, XGBoost ou Gradient Boosting)
   - Ajuste de hiperparâmetros e avaliação de desempenho

3. **Avaliação do Modelo**: Analisa o desempenho do modelo com métricas relevantes:
   - R² (Coeficiente de determinação)
   - MAE (Erro médio absoluto)
   - RMSE (Raiz do erro quadrático médio)
   - MAPE (Erro percentual absoluto médio)

4. **Geração de Previsões**: Aplica o modelo para prever demandas futuras:
   - Preparação dos dados existentes para previsão
   - Aplicação do modelo treinado
   - Interpretação e formatação dos resultados

5. **Recomendação de Produtos**: Analisa os produtos existentes e recomenda quais comprar:
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
```

## Detalhes do Algoritmo

### Engenharia de Características

O algoritmo realiza transformações sofisticadas nos dados:

- Extração de features temporais (mês, dia, dia da semana)
- Codificação cíclica para variáveis sazonais
- One-hot encoding para variáveis categóricas
- Normalização de variáveis numéricas
- Criação de features de interação

### Modelos Suportados

O sistema suporta múltiplos algoritmos de aprendizado de máquina:

- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor
- XGBoost (quando disponível)

### Visualizações

O sistema gera visualizações para facilitar a interpretação:

- Importância das features
- Comparação entre valores reais e previstos
- Distribuição dos erros de previsão
- Gráficos de recomendações de produtos

## Personalização

O algoritmo pode ser personalizado de várias maneiras:

1. **Algoritmo de Aprendizado**: Escolher entre diferentes modelos
2. **Engenharia de Características**: Adicionar ou modificar features
3. **Hiperparâmetros**: Ajustar parâmetros para otimizar o desempenho
4. **Estratégia de Recomendação**: Modificar os critérios de priorização

## Limitações Conhecidas

- Depende da qualidade e quantidade dos dados históricos
- Requer retreinamento periódico para capturar novas tendências
- Sensível a mudanças bruscas no mercado não refletidas nos dados

## Extensões Futuras

- Implementação de técnicas de deep learning para séries temporais
- Integração com fontes externas de dados (clima, economia)
- Interface de usuário para exploração interativa dos resultados
- Previsão automática e agendada para recomendações contínuas
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
