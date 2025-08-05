#!/bin/bash

# --- Definições de Cores para o Terminal ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sem Cor

# --- Variáveis de Configuração ---
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_MAIN_SCRIPT="main.py"

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Iniciando Script de Setup e Execução       ${NC}"
echo -e "${GREEN}=============================================${NC}"

# 1. Verificar se o comando python3 existe
if ! command -v python3 &> /dev/null
then
    echo -e "${RED}ERRO: python3 não encontrado no sistema.${NC}"
    echo -e "${RED}Por favor, instale o Python 3 para continuar.${NC}"
    exit 1
fi
echo -e "✅ Python 3 encontrado."

# 2. Verificar se o ambiente virtual existe. Se não, criar.
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Ambiente virtual ('$VENV_DIR') não encontrado. Criando...${NC}"
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERRO: Falha ao criar o ambiente virtual.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Ambiente virtual criado com sucesso.${NC}"
fi

# 3. Ativar o ambiente virtual
echo -e "Ativando o ambiente virtual..."
source $VENV_DIR/bin/activate
echo -e "${GREEN}✅ Ambiente virtual ativado.${NC}"

# 4. Instalar/Verificar dependências do requirements.txt
echo -e "Verificando/Instalando dependências do projeto (pip)..."
pip install -r $REQUIREMENTS_FILE
if [ $? -ne 0 ]; then
    echo -e "${RED}ERRO: Falha ao instalar as dependências do pip.${NC}"
    # Desativar o venv em caso de erro para não deixar o terminal "sujo"
    deactivate
    exit 1
fi
echo -e "${GREEN}✅ Dependências do projeto estão em dia.${NC}"

# 5. Executar o script principal do Python
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Setup concluído. Executando o programa...  ${NC}"
echo -e "${GREEN}=============================================${NC}"

# O "$@" passa todos os argumentos que foram dados ao run.sh para o script python
python $PYTHON_MAIN_SCRIPT "$@"

# Opcional: Desativar o venv ao final da execução
# deactivate
