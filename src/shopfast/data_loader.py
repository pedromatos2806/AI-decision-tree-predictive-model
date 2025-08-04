import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataLoader:
    """Classe para carregar e preparar dados para o modelo de previsão de demanda."""
    
    def __init__(self, file_path=None, delimiter=','):
        self.file_path = file_path
        self.delimiter = delimiter
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.product_ids_test = None
        self.preprocessor = None
        
    def load_data(self, required_columns=None, encoding='utf-8'):
        """
        Carrega os dados do arquivo.
        
        Args:
            required_columns: Lista de colunas obrigatórias
            encoding: Codificação do arquivo (default: utf-8)
        """
        if not self.file_path:
            raise ValueError("Caminho do arquivo não especificado.")
        
        # Carregamento dos dados
        try:
            # Tentar diferentes codificações se necessário
            encodings = [encoding, 'latin1', 'ISO-8859-1', 'cp1252']
            
            for enc in encodings:
                try:
                    # Tentar carregar o arquivo com o delimitador fornecido e tratando linhas com erro
                    self.data = pd.read_csv(
                        self.file_path, 
                        sep=self.delimiter,
                        encoding=enc,
                        on_bad_lines='skip'  # Pular linhas com problemas
                    )
                    print(f"Dados carregados com sucesso usando codificação {enc}. Formato: {self.data.shape}")
                    break
                except UnicodeDecodeError:
                    print(f"Erro de codificação com {enc}, tentando próxima...")
                    continue
            
            # Se ainda não carregou os dados, tentar sem especificar codificação
            if self.data is None:
                self.data = pd.read_csv(self.file_path, sep=self.delimiter, on_bad_lines='skip')
                print(f"Dados carregados com codificação padrão. Formato: {self.data.shape}")
            
            print(f"Colunas encontradas: {list(self.data.columns)}")
            
            # Limpar nomes das colunas (remover espaços extras)
            self.data.columns = self.data.columns.str.strip()
            
            # Remover linhas com datas inválidas ou incompletas
            if 'data' in self.data.columns:
                # Filtra as linhas onde 'data' tem o formato correto YYYY-MM-DD
                import re
                date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
                mask = self.data['data'].astype(str).apply(lambda x: bool(date_pattern.match(x)))
                
                invalid_rows = self.data[~mask]
                if len(invalid_rows) > 0:
                    print(f"Removendo {len(invalid_rows)} linhas com datas inválidas.")
                    self.data = self.data[mask]
            
            # Definir colunas obrigatórias
            if required_columns is None:
                required_columns = [
                    'data', 'produto_id', 'categoria', 'quantidade_vendida',
                    'preco_unitario', 'promocao', 'temperatura_media',
                    'humidade_media', 'dia_da_semana'
                ]
            
            # Verificar colunas existentes e ausentes
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                print(f"AVISO: As seguintes colunas obrigatórias não foram encontradas: {missing_columns}")
                print("Tentando carregar o arquivo com configurações alternativas...")
                
                # Tentar carregar com diferentes delimitadores se houver problemas
                for alt_delimiter in [',', ';', '\t', '|']:
                    if alt_delimiter == self.delimiter:
                        continue
                        
                    print(f"Tentando delimitador: '{alt_delimiter}'")
                    try:
                        alt_data = pd.read_csv(self.file_path, sep=alt_delimiter, on_bad_lines='skip')
                        alt_missing = [col for col in required_columns if col not in alt_data.columns]
                        
                        if len(alt_missing) < len(missing_columns):
                            print(f"Melhor resultado com delimitador '{alt_delimiter}'")
                            self.data = alt_data
                            self.delimiter = alt_delimiter
                            print(f"Colunas encontradas: {list(self.data.columns)}")
                            missing_columns = alt_missing
                    except Exception as e:
                        print(f"Erro com delimitador '{alt_delimiter}': {str(e)}")
                
                # Se ainda faltarem colunas, prosseguir com aviso
                if missing_columns:
                    print(f"AVISO: As seguintes colunas ainda estão faltando: {missing_columns}")
                    print("Continuando o processamento sem estas colunas.")
            
            return self.data
        except Exception as e:
            print(f"Erro ao carregar os dados: {str(e)}")
            print("Verifique se o arquivo está no formato correto e tente novamente.")
            raise
    
    def preprocess_data(self, target_column='quantidade_vendida', test_size=0.2, random_state=42):
        """Prepara os dados para treinamento e teste."""
        if self.data is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")
        
        # Verificar se a coluna alvo existe
        if target_column not in self.data.columns:
            raise ValueError(f"Coluna alvo '{target_column}' não encontrada nos dados.")
        
        # Fazer uma cópia para evitar avisos de SettingWithCopyWarning
        self.data = self.data.copy()
        
        try:
            # Converter data para timestamp com tratamento de erros
            try:
                self.data['data'] = pd.to_datetime(self.data['data'], errors='coerce')
                # Remover linhas onde a conversão da data falhou
                invalid_dates = self.data['data'].isna().sum()
                if invalid_dates > 0:
                    print(f"Removendo {invalid_dates} linhas com datas inválidas.")
                    self.data = self.data.dropna(subset=['data'])
            except Exception as e:
                print(f"Erro ao converter datas: {str(e)}. Tentando abordagem alternativa...")
                # Abordagem alternativa: remover a coluna data do modelo
                print("Removendo a coluna 'data' do modelo.")
                self.data['data_valida'] = False
            
            # Extrair características de data se a conversão foi bem-sucedida
            if 'data_valida' not in self.data.columns:
                self.data['mes'] = self.data['data'].dt.month
                self.data['dia'] = self.data['data'].dt.day
            
            # Converter 'promocao' para binário se existir
            if 'promocao' in self.data.columns:
                # Normalizar valores para comparação
                self.data['promocao'] = self.data['promocao'].astype(str).str.lower()
                # Considerar várias formas de "Sim"
                self.data['promocao'] = self.data['promocao'].apply(
                    lambda x: 1 if x in ['sim', 's', 'yes', 'y', '1', 'true'] else 0
                )
            
            # Definir features e target, excluindo colunas não utilizáveis
            exclude_cols = [target_column]
            if 'data_valida' not in self.data.columns:
                exclude_cols.append('data')
            
            X = self.data.drop(exclude_cols, axis=1)
            y = self.data[target_column]
            
            # Separar os dados de treinamento e teste
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Guardar IDs dos produtos para avaliação se existir
            if 'produto_id' in self.X_test.columns:
                self.product_ids_test = self.X_test['produto_id'].copy()
                
                # Remover ID de produto das features
                self.X_train = self.X_train.drop(['produto_id'], axis=1)
                self.X_test = self.X_test.drop(['produto_id'], axis=1)
            
            # Definir colunas numéricas e categóricas
            numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = self.X_train.select_dtypes(include=['object']).columns
            
            # Criar preprocessadores
            transformers = []
            
            if len(numerical_cols) > 0:
                numerical_transformer = Pipeline(steps=[
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', numerical_transformer, numerical_cols))
            
            if len(categorical_cols) > 0:
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                transformers.append(('cat', categorical_transformer, categorical_cols))
            
            # Combinar preprocessadores
            self.preprocessor = ColumnTransformer(transformers=transformers)
            
            # Aplicar transformações
            self.X_train = self.preprocessor.fit_transform(self.X_train)
            self.X_test = self.preprocessor.transform(self.X_test)
            
            print(f"Dados pré-processados. X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test, self.product_ids_test
        except Exception as e:
            print(f"Erro durante o pré-processamento: {str(e)}")
            raise
        
    def get_feature_names(self):
        """Retorna os nomes das features após transformação."""
        if self.preprocessor is None:
            raise ValueError("Preprocessador não inicializado. Execute preprocess_data() primeiro.")
            
        # Obter nomes das colunas originais
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Remover colunas não usadas
        if 'produto_id' in numerical_cols:
            numerical_cols = [col for col in numerical_cols if col != 'produto_id']
            
        # Obter nomes das colunas transformadas
        feature_names = []
        
        # Adicionar colunas numéricas
        for col in numerical_cols:
            if col != 'quantidade_vendida':  # Excluir a coluna alvo
                feature_names.append(col)
                
        # Adicionar colunas categóricas (one-hot encoded)
        for col in categorical_cols:
            unique_values = self.data[col].unique()
            for val in unique_values:
                feature_names.append(f"{col}_{val}")
                
        return feature_names