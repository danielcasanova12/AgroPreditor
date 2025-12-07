# -*- coding: utf-8 -*-
"""
04_create_master_datasets.py

Este script é o quarto e um dos mais importantes passos no pipeline.
Ele é responsável por carregar os dados preparados de clima, NDVI e produtividade,
juntá-los e criar os três datasets mestres para a modelagem: diário, mensal e anual.

Passos executados:
1.  Carrega os arquivos pré-processados:
    - 'clima_PR_safra_season.csv' (dados climáticos da safra)
    - 'NDVI_Municipios_filtrado.csv' (dados de NDVI)
    - 'soja_produtividade_calculada.csv' (dados de produtividade)
2.  Define uma função para mapear cada data a uma 'SAFRA' (ex: '00/01').
3.  Aplica o mapeamento de safra aos dataframes de clima e NDVI.
4.  Filtra todos os datasets para garantir que apenas os municípios de interesse sejam incluídos.
5.  Junta os três datasets, criando uma base diária (`df_daily`).
    - O NDVI (mensal) e a produtividade (anual) são replicados para cada dia correspondente.
6.  A partir da base diária, cria agregações para os níveis mensal e anual.
    - A lógica de agregação é customizada (ex: 'sum' para chuva, 'mean' para temperatura).
7.  Salva os três datasets finais:
    - 'master_diario.csv'
    - 'master_mensal.csv'
    - 'master_anual.csv'
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

def date_to_safra(d):
    """
    Converte uma data para o formato de safra 'YY/YY+1'.
    A safra da soja no Brasil começa no segundo semestre de um ano e termina no primeiro do ano seguinte.
    Ex: data em 2000-10 -> safra '00/01'
        data em 2001-01 -> safra '00/01'
    """
    y = d.year
    m = d.month
    if m >= 9:  # Setembro a Dezembro: início da safra
        ano_ini = y
    else:       # Janeiro a Março: fim da safra
        ano_ini = y - 1
    ano_fim = ano_ini + 1
    return f"{ano_ini % 100:02d}/{ano_fim % 100:02d}"

def create_master_datasets(clima_path, ndvi_path, prod_path, output_dir, municipios_interesse):
    """
    Junta os dados e cria os datasets mestres (diário, mensal, anual).
    """
    logging.info("Iniciando a criação dos datasets mestres...")

    try:
        # 1. Carregar bases pré-processadas
        df_clima = pd.read_csv(clima_path)
        df_ndvi = pd.read_csv(ndvi_path)
        df_prod = pd.read_csv(prod_path)
        logging.info("Arquivos de clima, NDVI e produtividade carregados.")
    except FileNotFoundError as e:
        logging.error(f"Erro ao carregar arquivos de entrada: {e}")
        return

    # --- Preparação e Padronização ---
    # Clima: renomear colunas e converter data
    df_clima.rename(columns={'Data': 'data', 'Municipio': 'municipio'}, inplace=True)
    df_clima['data'] = pd.to_datetime(df_clima['data'], errors='coerce')
    df_clima.dropna(subset=['data'], inplace=True)
    df_clima['municipio'] = df_clima['municipio'].str.strip()

    # NDVI: converter data
    df_ndvi['data'] = pd.to_datetime(df_ndvi['data'], errors='coerce')
    df_ndvi.dropna(subset=['data'], inplace=True)
    df_ndvi['municipio'] = df_ndvi['municipio'].str.strip()

    # Produção: garantir que 'municipio' está limpo
    df_prod['municipio'] = df_prod['municipio'].str.strip()

    # 2. Mapear data para SAFRA
    df_clima['SAFRA'] = df_clima['data'].apply(date_to_safra)
    df_ndvi['SAFRA'] = df_ndvi['data'].apply(date_to_safra)
    logging.info("Coluna 'SAFRA' criada para dados de clima e NDVI.")

    # 3. Filtrar todos os dataframes pelos municípios de interesse
    df_clima = df_clima[df_clima['municipio'].isin(municipios_interesse)].copy()
    df_ndvi = df_ndvi[df_ndvi['municipio'].isin(municipios_interesse)].copy()
    df_prod = df_prod[df_prod['municipio'].isin(municipios_interesse)].copy()
    logging.info(f"Dataframes filtrados. Clima: {len(df_clima)}, NDVI: {len(df_ndvi)}, Produção: {len(df_prod)} linhas.")

    # --- Criação do Dataset Diário ---
    logging.info("Iniciando junção para o dataset diário...")

    # 4. Pré-agregar NDVI para nível mensal para evitar duplicatas
    df_ndvi['ano'] = df_ndvi['data'].dt.year
    df_ndvi['mes'] = df_ndvi['data'].dt.month
    df_ndvi_mensal = (
        df_ndvi
        .groupby(['municipio', 'SAFRA', 'ano', 'mes'], as_index=False)['valor']
        .mean()
        .rename(columns={'valor': 'NDVI'})
    )

    # 5. Juntar clima diário com NDVI mensal
    df_clima['ano'] = df_clima['data'].dt.year
    df_clima['mes'] = df_clima['data'].dt.month
    df_daily = pd.merge(
        df_clima,
        df_ndvi_mensal,
        on=['municipio', 'SAFRA', 'ano', 'mes'],
        how='left'
    )

    # 6. Juntar com dados de produção (anuais por safra)
    df_daily = pd.merge(
        df_daily,
        df_prod,
        on=['SAFRA', 'municipio'],
        how='left',
        suffixes=('', '_prod') # Adiciona sufixo para colunas duplicadas como REGIAO, Solo
    )
    # Se houver colunas duplicadas, priorizar as do df_prod que são anuais
    if 'REGIAO_prod' in df_daily.columns:
        df_daily['REGIAO'] = df_daily['REGIAO_prod']
        df_daily.drop(columns=['REGIAO_prod', 'REGIAO_x'], errors='ignore', inplace=True)
    if 'Solo_prod' in df_daily.columns:
        df_daily['Solo'] = df_daily['Solo_prod']
        df_daily.drop(columns=['Solo_prod', 'Solo_x'], errors='ignore', inplace=True)


    # 7. Limpeza final do dataset diário
    # Manter apenas linhas onde temos dados de NDVI e Produção
    final_cols = ['NDVI', 'AREA TOTAL', 'PRODUCAO', 'YIELD_SC_HA']
    df_daily.dropna(subset=final_cols, inplace=True)
    logging.info(f"Dataset diário criado com {len(df_daily)} linhas.")

    # Salvar dataset diário
    daily_out = os.path.join(output_dir, 'master_diario.csv')
    df_daily.to_csv(daily_out, index=False)
    logging.info(f"Dataset diário salvo em: {daily_out}")

    # --- Criação dos Datasets Agregados ---
    climate_cols = [
        'Altitude (m)', 'Tmax (°C)', 'Tmin (°C)', 'Tmed (°C)',
        'UR (%)', 'U2 (m/s)', 'RS (MJ/m²d)', 'Chuva (mm)'
    ]
    # Garantir que as colunas existem antes de agregar
    climate_cols = [col for col in climate_cols if col in df_daily.columns]

    # 8. Criar tabela MENSAL
    logging.info("Criando dataset mensal...")
    group_cols_m = ['municipio', 'SAFRA', 'ano', 'mes', 'REGIAO', 'Solo']
    agg_dict_m = {c: 'mean' for c in climate_cols}
    agg_dict_m['Chuva (mm)'] = 'sum'
    agg_dict_m['NDVI'] = 'mean'
    agg_dict_m['AREA TOTAL'] = 'first'
    agg_dict_m['PRODUCAO'] = 'first'
    agg_dict_m['YIELD_SC_HA'] = 'first'

    df_monthly = df_daily.groupby(group_cols_m, as_index=False).agg(agg_dict_m)
    monthly_out = os.path.join(output_dir, 'master_mensal.csv')
    df_monthly.to_csv(monthly_out, index=False)
    logging.info(f"Dataset mensal salvo em: {monthly_out}")

    # 9. Criar tabela ANUAL (por SAFRA)
    logging.info("Criando dataset anual...")
    group_cols_a = ['municipio', 'SAFRA', 'REGIAO', 'Solo']
    agg_dict_a = {c: 'mean' for c in climate_cols}
    agg_dict_a['Chuva (mm)'] = 'sum'
    agg_dict_a['NDVI'] = 'mean'
    agg_dict_a['AREA TOTAL'] = 'first'
    agg_dict_a['PRODUCAO'] = 'first'
    agg_dict_a['YIELD_SC_HA'] = 'first'

    df_annual = df_daily.groupby(group_cols_a, as_index=False).agg(agg_dict_a)
    annual_out = os.path.join(output_dir, 'master_anual.csv')
    df_annual.to_csv(annual_out, index=False)
    logging.info(f"Dataset anual salvo em: {annual_out}")


if __name__ == '__main__':
    # Importar as configurações
    import config

    # Cria o diretório de saída se não existir
    os.makedirs(config.MASTER_DATA_DIR, exist_ok=True)

    # Executa a função principal usando os caminhos e parâmetros do config
    create_master_datasets(
        clima_path=config.CLIMATE_PROCESSED_PATH,
        ndvi_path=config.NDVI_PROCESSED_PATH,
        prod_path=config.YIELD_PROCESSED_PATH,
        output_dir=config.MASTER_DATA_DIR,
        municipios_interesse=config.MUNICIPIOS_DE_INTERESSE
    )
