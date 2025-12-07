# -*- coding: utf-8 -*-
"""
02_prepare_yield.py

Este script é o segundo passo no pipeline de processamento de dados.
Ele é responsável por carregar os dados de produção de soja e calcular a produtividade.

Passos executados:
1.  Carrega o arquivo de dados de produção ('soja_por_ano_municipio_area.csv').
2.  Converte as colunas 'PRODUCAO' (toneladas) e 'AREA TOTAL' (hectares) para formato numérico.
3.  Calcula a produtividade (YIELD) em sacas por hectare.
    - Fator de conversão: 1 tonelada = 1000 kg; 1 saca = 60 kg.
    - Fórmula: YIELD = (PRODUCAO_ton * 1000 / 60) / AREA_ha
4.  Renomeia e seleciona as colunas finais.
5.  Salva o resultado em um novo arquivo CSV ('soja_produtividade_calculada.csv').
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

def prepare_yield_data(input_path, output_path):
    """
    Calcula a produtividade da soja a partir dos dados de produção e área.

    Args:
        input_path (str): Caminho para o arquivo CSV de entrada.
        output_path (str): Caminho para salvar o arquivo CSV processado.
    """
    logging.info("Iniciando a preparação dos dados de produtividade...")

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Arquivo {input_path} carregado com {len(df)} linhas.")
    except FileNotFoundError:
        logging.error(f"Arquivo de entrada não encontrado em: {input_path}")
        return

    # Validar a existência das colunas necessárias
    required_cols = ["PRODUCAO", "AREA TOTAL", "Município", "SAFRA", "REGIAO"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logging.error(f"Colunas obrigatórias não encontradas no arquivo: {missing}")
        return

    # Converter colunas para numérico, tratando vírgulas como separador decimal
    for col in ["PRODUCAO", "AREA TOTAL"]:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remover linhas onde a conversão falhou ou a área é zero
    df.dropna(subset=["PRODUCAO", "AREA TOTAL"], inplace=True)
    df = df[df["AREA TOTAL"] > 0]

    # Fator de conversão: toneladas -> sacas (1 saca = 60 kg)
    fator_ton_para_sacas = 1000 / 60

    # Calcular produtividade em sacas por hectare
    df["YIELD_SC_HA"] = (df["PRODUCAO"] * fator_ton_para_sacas) / df["AREA TOTAL"]
    df["YIELD_SC_HA"] = df["YIELD_SC_HA"].round(2)
    logging.info(f"Coluna 'YIELD_SC_HA' calculada. Média: {df['YIELD_SC_HA'].mean():.2f} sc/ha.")

    # Renomear e selecionar colunas para o arquivo final
    df.rename(columns={"Município": "municipio"}, inplace=True)
    df['municipio'] = df['municipio'].str.strip()

    # Manter as colunas originais de produção e área para referência futura, se necessário
    output_cols = [
        "municipio", "SAFRA", "REGIAO",
        "AREA TOTAL", "PRODUCAO", "YIELD_SC_HA"
    ]
    df_final = df[output_cols]

    # Salvar o arquivo processado
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        logging.info(f"Dados de produtividade preparados e salvos em: {output_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo em {output_path}: {e}")

if __name__ == '__main__':
    # Importar as configurações
    import config

    # Criar o diretório de saída se ele não existir
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Executa a função principal usando os caminhos do arquivo de configuração
    prepare_yield_data(
        input_path=config.YIELD_RAW_PATH,
        output_path=config.YIELD_PROCESSED_PATH
    )
