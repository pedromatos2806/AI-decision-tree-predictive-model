# ShopFast - Previsão de Demanda com IA

## Problema

A ShopFast, um e-commerce de vários produtos, tinha dificuldade em prever a demanda de cada item, causando falta ou excesso de estoque. O objetivo era usar Inteligência Artificial para prever a demanda de cada produto, melhorar o estoque e aumentar a satisfação do cliente.

## Solução

Desenvolvi um sistema de previsão de demanda usando Machine Learning, que analisa os dados históricos de vendas e fatores como preço, promoções, clima e dia da semana. O sistema indica quanto de cada produto deve ser comprado.

## Como Funciona o Algoritmo

1. **Leitura e Limpeza dos Dados**: Carrega o arquivo de vendas, trata valores ausentes e outliers.
2. **Engenharia de Atributos**: Cria novas colunas úteis, como mês, dia, se é fim de semana, etc.
3. **Normalização e Codificação**: Prepara os dados para o modelo (padroniza números e transforma categorias em números).
4. **Divisão dos Dados**: Separa em treino (80%) e teste (20%) respeitando a ordem do tempo.
5. **Treinamento**: Testa vários algoritmos (Random Forest, XGBoost, etc.) e escolhe o melhor usando validação cruzada temporal.
6. **Avaliação**: Mede o desempenho com métricas como R², MAE, RMSE e MAPE.
7. **Previsão e Recomendação**: Faz previsões para novos dados e recomenda quais produtos comprar, priorizando os mais importantes.

## Principais Técnicas de IA Utilizadas

- **Random Forest/XGBoost**: Modelos de árvore para prever a quantidade vendida.
- **Validação Cruzada Temporal**: Garante que o modelo não "vê o futuro".
- **Engenharia de Atributos**: Criação de variáveis temporais e interações.
- **Grid Search**: Busca automática dos melhores parâmetros do modelo.

## Variáveis Usadas

- Data, Produto, Categoria, Preço, Promoção, Temperatura, Umidade, Dia da Semana, Feedback do Cliente.

## Como Usar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o treinamento e a recomendação:
   ```bash
   python main.py
   ```
3. Para prever novos produtos:
   ```bash
   python fazer_previsao.py
   ```

## Resultados

O modelo atinge R² entre 0.85 e 0.92, com erro médio baixo. Gera gráficos de importância das variáveis, comparação real vs previsto e recomendações de compra.

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

## Resumo para Apresentação

- O sistema lê os dados, prepara, treina o modelo, avalia e recomenda compras.
- Usa IA para ajudar a empresa a comprar melhor e evitar prejuízos.
- Fácil de rodar e adaptar para outros cenários.
