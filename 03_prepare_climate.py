# -*- coding: utf-8 -*-
"""
03_prepare_climate.py

Este script é o terceiro passo no pipeline de processamento de dados.
Ele é responsável por preparar os dados climáticos, filtrando-os para o período
relevante da safra da soja.

Passos executados:
1.  Carrega o arquivo de dados climáticos brutos ('clima_PR_2000-2024_clean.csv').
2.  Remove colunas que não serão utilizadas (ex: Latitude, Longitude) e colunas totalmente nulas.
3.  Converte a coluna de data para o formato datetime.
4.  Filtra os dados para manter apenas os meses da safra da soja (setembro a março).
5.  Salva o resultado em um novo arquivo CSV ('clima_PR_safra_season.csv').
"""
import pandas as pd
import logging
import os

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution.log", mode='a'),
        logging.StreamHandler()
    ]
)

def prepare_climate_data(input_path, output_path, meses_safra):
    """
    Filtra e prepara os dados climáticos para o período da safra.

    Args:
        input_path (str): Caminho para o arquivo CSV de entrada.
        output_path (str): Caminho para salvar o arquivo CSV processado.
        meses_safra (list): Lista de inteiros representando os meses da safra.
    """
    logging.info("Iniciando a preparação dos dados climáticos...")

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Arquivo {input_path} carregado com {len(df)} linhas.")
    except FileNotFoundError:
        logging.error(f"Arquivo de entrada não encontrado em: {input_peth}")
        return

    # 1. Remover colunas totalmente nulas e as de geolocalização
    df.dropna(axis=1, how='all', inplace=True)
    df.drop(columns=['Latitude', 'Longitude'], errors='ignore', inplace=True)
    logging.info("Colunas nulas e de geolocalização removidas.")

    # 2. Converter a coluna de data
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    # Remover linhas onde a conversão de data falhou
    df.dropna(subset=['Data'], inplace=True)

    # 3. Filtrar pelos meses da safra
    df_filtrado = df[df['Data'].dt.month.isin(meses_safra)].copy()
    logging.info(f"Filtragem por meses da safra resultou em {len(df_filtrado)} linhas.")

    # 4. Salvar o arquivo processado
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_filtrado.to_csv(output_path, index=False)
        logging.info(f"Dados climáticos preparados e salvos em: {output_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo em {output_path}: {e}")

if __name__ == '__main__':
    # Importar as configurações
    import config

    # Criar o diretório de saída se ele não existir
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Executa a função principal usando os caminhos e parâmetros do arquivo de configuração
    prepare_climate_data(
        input_path=config.CLIMATE_RAW_PATH,
        output_path=config.CLIMATE_PROCESSED_PATH,
        meses_safra=config.MESES_DA_SAFRA
    )
