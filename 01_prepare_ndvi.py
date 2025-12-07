# -*- coding: utf-8 -*-
"""
01_prepare_ndvi.py

Este script é o primeiro passo no pipeline de processamento de dados.
Ele é responsável por limpar e preparar os dados de NDVI (Normalized Difference Vegetation Index).

Passos executados:
1.  Carrega o arquivo de dados brutos de NDVI ('NDVI_Municipios_unico.csv').
2.  Extrai e limpa os nomes dos municípios a partir de uma string complexa.
3.  Filtra os dados para manter apenas os municípios de interesse para a análise.
4.  Salva o resultado em um novo arquivo CSV ('NDVI_Municipios_filtrado.csv'), pronto para ser usado nas próximas etapas.
"""
import pandas as pd
import logging
import os

# Configuração do logging para registrar informações sobre a execução
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution.log", mode='a'),
        logging.StreamHandler()
    ]
)

def prepare_ndvi_data(input_path, output_path, municipios_interesse):
    """
    Limpa e filtra os dados de NDVI.

    Args:
        input_path (str): Caminho para o arquivo CSV de entrada.
        output_path (str): Caminho para salvar o arquivo CSV processado.
        municipios_interesse (list): Lista de nomes de municípios a serem mantidos.
    """
    logging.info("Iniciando a preparação dos dados de NDVI...")

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Arquivo {input_path} carregado com {len(df)} linhas.")
    except FileNotFoundError:
        logging.error(f"Arquivo de entrada não encontrado em: {input_path}")
        return

    # 1. Extrair nome do município da coluna 'municipio'
    # Ex: "api_Municipios — camada_unida_Pinhal_de_São_Bento_1_1" -> "Pinhal_de_São_Bento"
    df['municipio_raw'] = df['municipio'].str.extract(
        r'unida_(.+?)_\d+_\d+',
        expand=False
    )

    # Se o padrão não for encontrado, usa o valor original como fallback
    df['municipio_raw'] = df['municipio_raw'].fillna(df['municipio'])

    # 2. Limpar o nome do município (trocar "_" por espaço)
    # Ex: "Pinhal_de_São_Bento" -> "Pinhal de São Bento"
    df['municipio_nome'] = (
        df['municipio_raw']
        .str.replace('_', ' ', regex=False)
        .str.strip()
    )

    # 3. Filtrar pelos municípios de interesse
    df_filtrado = df[df['municipio_nome'].isin(municipios_interesse)].copy()
    logging.info(f"Filtragem por municípios de interesse resultou em {len(df_filtrado)} linhas.")

    if df_filtrado.empty:
        logging.warning("Nenhum município de interesse foi encontrado no arquivo de NDVI. O arquivo de saída estará vazio.")

    # 4. Definir a coluna final 'municipio' e selecionar colunas
    df_filtrado['municipio'] = df_filtrado['municipio_nome']
    df_final = df_filtrado[['data', 'valor', 'municipio']]

    # 5. Salvar o arquivo processado
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False, float_format='%.4f')
        logging.info(f"Dados de NDVI preparados e salvos em: {output_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo em {output_path}: {e}")

if __name__ == '__main__':
    # Importar as configurações
    import config

    # Criar o diretório de saída se ele não existir
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Executa a função principal usando os caminhos do arquivo de configuração
    prepare_ndvi_data(
        input_path=config.NDVI_RAW_PATH,
        output_path=config.NDVI_PROCESSED_PATH,
        municipios_interesse=config.MUNICIPIOS_DE_INTERESSE
    )
