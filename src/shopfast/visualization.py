import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ModelVisualizer:
    """Classe para visualização de resultados do modelo de previsão de demanda."""
    
    def __init__(self, output_dir=None):
        """
        Inicializa o visualizador.
        
        Args:
            output_dir: Diretório onde os gráficos serão salvos.
        """
        self.output_dir = output_dir
        plt.style.use('ggplot')
        
    def plot_feature_importance(self, feature_importance, top_n=20, figsize=(12, 8)):
        """
        Plota a importância das features.
        
        Args:
            feature_importance: DataFrame com colunas 'Feature' e 'Importance'
            top_n: Número de features principais para mostrar
            figsize: Tamanho da figura
        """
        if feature_importance is None:
            raise ValueError("Feature importance não fornecida")
            
        plt.figure(figsize=figsize)
        
        # Selecionar as top_n features
        top_features = feature_importance.head(top_n)
        
        # Criar gráfico de barras horizontais
        plt.barh(
            range(len(top_features)), 
            top_features['Importance'],
            align='center'
        )
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.title('Importância das Features na Previsão de Demanda', fontsize=15)
        plt.xlabel('Importância')
        plt.tight_layout()
        
        # Salvar gráfico se o diretório de saída for fornecido
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            
        return plt.gcf()
        
    def plot_actual_vs_predicted(self, y_test, y_pred, figsize=(10, 6)):
        """
        Plota valores reais vs. previstos.
        
        Args:
            y_test: Valores reais
            y_pred: Valores previstos
            figsize: Tamanho da figura
        """
        plt.figure(figsize=figsize)
        
        # Adicionar linha diagonal de referência (previsão perfeita)
        max_val = max(np.max(y_test), np.max(y_pred))
        min_val = min(np.min(y_test), np.min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        # Plotar valores reais vs. previstos
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        plt.title('Valores Reais vs. Previstos', fontsize=15)
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.tight_layout()
        
        # Salvar gráfico se o diretório de saída for fornecido
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
            
        return plt.gcf()
        
    def plot_prediction_results(self, prediction_df, sample_size=30, figsize=(12, 6)):
        """
        Plota uma comparação entre demanda real e prevista para uma amostra de produtos.
        
        Args:
            prediction_df: DataFrame com colunas 'produto_id', 'demanda_real' e 'demanda_prevista'
            sample_size: Tamanho da amostra a ser visualizada
            figsize: Tamanho da figura
        """
        if len(prediction_df) > sample_size:
            # Usar uma amostra se o DataFrame for grande demais
            sample_df = prediction_df.sample(sample_size, random_state=42)
        else:
            sample_df = prediction_df.copy()
            
        plt.figure(figsize=figsize)
        
        # Reordenar para melhor visualização
        sample_df = sample_df.sort_values('demanda_real', ascending=False)
        
        # Criar gráfico de barras agrupadas
        x = np.arange(len(sample_df))
        width = 0.35
        
        plt.bar(x - width/2, sample_df['demanda_real'], width, label='Real', alpha=0.7)
        plt.bar(x + width/2, sample_df['demanda_prevista'], width, label='Prevista', alpha=0.7)
        
        plt.xlabel('Produtos')
        plt.ylabel('Quantidade')
        plt.title('Demanda Real vs. Prevista por Produto', fontsize=15)
        plt.xticks(x, sample_df['produto_id'], rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # Salvar gráfico se o diretório de saída for fornecido
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/prediction_results.png", dpi=300, bbox_inches='tight')
            
        return plt.gcf()
        
    def plot_error_distribution(self, y_test, y_pred, figsize=(10, 6)):
        """
        Plota a distribuição dos erros de previsão.
        
        Args:
            y_test: Valores reais
            y_pred: Valores previstos
            figsize: Tamanho da figura
        """
        errors = y_test - y_pred
        
        plt.figure(figsize=figsize)
        
        # Histograma dos erros
        sns.histplot(errors, kde=True)
        
        plt.title('Distribuição dos Erros de Previsão', fontsize=15)
        plt.xlabel('Erro (Real - Previsto)')
        plt.ylabel('Frequência')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        
        # Salvar gráfico se o diretório de saída for fornecido
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
            
        return plt.gcf()
