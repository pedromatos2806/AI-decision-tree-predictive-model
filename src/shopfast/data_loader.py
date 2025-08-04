import pandas as pd

class DataLoader:
    """Responsável por carregar e limpar os dados brutos."""
    def __init__(self, file_path, delimiter=','):
        self.file_path = file_path
        self.delimiter = delimiter

    def load_and_clean_data(self):
        """Carrega dados do arquivo, limpa nomes de colunas e remove dados ausentes."""
        try:
            df = pd.read_csv(self.file_path, delimiter=self.delimiter)
            print(f"Arquivo '{self.file_path}' carregado com sucesso.")
            
            # Limpeza de nomes das colunas (remove espaços)
            df.columns = df.columns.str.strip()
            
            # Remove linhas com qualquer valor ausente
            df.dropna(inplace=True)
            print("Limpeza de dados concluída.")
            return df
        except FileNotFoundError:
            print(f"ERRO: O arquivo '{self.file_path}' não foi encontrado.")
            return None