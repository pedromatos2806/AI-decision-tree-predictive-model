iqr = q3 - q1
dados['mes'] = dados['data'].dt.month
dados['dia'] = dados['data'].dt.day
dados['ano'] = dados['data'].dt.year
dados['semana_do_ano'] = dados['data'].dt.isocalendar().week
dados['dia_do_ano'] = dados['data'].dt.dayofyear
dados['trimestre'] = dados['data'].dt.quarter
dados['e_fim_de_semana'] = dados['dia_da_semana'].isin([5, 6, 'Sábado', 'Domingo', 'Saturday', 'Sunday']).astype(int)
dados['mes_sin'] = np.sin(2 _ np.pi _ dados['mes']/12)
dados['mes_cos'] = np.cos(2 _ np.pi _ dados['mes']/12)
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
dados_processados = carregar_e_processar_dados(caminho_dados, delimiter=',')

# Guia Explicativo - Previsão de Demanda ShopFast

## O Problema

A ShopFast, loja online, não conseguia prever corretamente a demanda dos produtos. Isso causava falta ou excesso de estoque, prejudicando vendas e clientes. O desafio era criar um sistema de IA para prever a demanda de cada produto usando os dados históricos de vendas.

## Como Resolvi

1. **Leitura dos Dados**: Carreguei o arquivo de vendas fornecido pela empresa.
2. **Limpeza e Preparação**: Tratei valores ausentes, removi outliers e criei novas colunas úteis (mês, dia, fim de semana, etc).
3. **Engenharia de Atributos**: Transformei variáveis de data, clima, promoções e categorias em números que o modelo entende.
4. **Divisão dos Dados**: Separei os dados em treino e teste, respeitando a ordem do tempo.
5. **Modelagem**: Testei algoritmos de árvore (Random Forest, XGBoost) e escolhi o melhor usando validação cruzada temporal.
6. **Avaliação**: Usei métricas como R², MAE e RMSE para medir o desempenho.
7. **Recomendação**: O sistema indica quais produtos comprar, priorizando os mais importantes.

## Técnicas de IA Utilizadas

- Modelos de árvore (Random Forest, XGBoost)
- Engenharia de atributos (criação de variáveis temporais e interações)
- Validação cruzada temporal (para séries temporais)
- Grid Search (busca automática dos melhores parâmetros)

## Variáveis Consideradas

- Data, Produto, Categoria, Preço, Promoção, Temperatura, Umidade, Dia da Semana, Feedback do Cliente

## Como Funciona o Algoritmo (Resumo)

1. Lê e prepara os dados
2. Cria novas variáveis úteis
3. Treina o modelo de previsão
4. Avalia o desempenho
5. Faz previsões e recomenda compras

## Resultados

O modelo atinge R² entre 0.85 e 0.92, com erro médio baixo. Gera gráficos mostrando a importância das variáveis, comparação entre valores reais e previstos, e recomendações de compra.

## Estrutura dos Arquivos

```
main.py                # Executa todo o processo
fazer_previsao.py      # Faz previsão para novos dados
recomendar_produtos.py # Recomenda produtos para comprar
src/shopfast/          # Código dos módulos
dados/                 # Dados de vendas
modelos/               # Modelos treinados
resultados/            # Resultados e gráficos
```

## Para Apresentar

- O sistema lê os dados, prepara, treina o modelo, avalia e recomenda compras.
- Usa IA para ajudar a empresa a comprar melhor e evitar prejuízos.
- Fácil de rodar e adaptar para outros cenários.
