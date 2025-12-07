# -*- coding: utf-8 -*-
"""
05_feature_engineering.py

Este script é o coração da análise, onde todo o conhecimento de domínio é
transformado em features para os modelos de machine learning.

Passos executados:
1.  Carrega os datasets mestres (diário, mensal, anual) criados na etapa anterior.
2.  Para cada dataset, aplica uma série de transformações para criar novas features:
    a.  **Features Agronômicas Básicas:** GDD (Growing Degree Days) e VPD (Vapor Pressure Deficit).
    b.  **Features de Estresse:** Contagem de dias com estresse térmico (Heat Stress) e dias secos.
    c.  **Features Acumuladas:** Janelas móveis (rolling windows) para chuva e GDD acumulados (ex: 30, 60, 90 dias).
    d.  **Features de Sinergia/Interação:** Interação entre NDVI e clima (ex: NDVI * Chuva).
    e.  **Features Polinomiais:** Termos quadráticos para capturar relações não-lineares (ex: temperatura ótima).
    f.  **Features de Defasagem (Lag):** Condições da safra anterior (ex: chuva e NDVI do ano anterior).
    g.  **Balanço Hídrico:** Tenta calcular uma feature de reserva hídrica do solo com base em dados climáticos anuais.
3.  Salva os datasets finais enriquecidos em um novo diretório ('./data/features/'), prontos para a modelagem.
"""
import pandas as pd
import numpy as np
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

def add_agronomic_features(df, is_daily=False):
    """
    Adiciona um conjunto rico de features agronômicas a um dataframe.
    O dataframe deve estar ordenado por município e data.
    """
    df_out = df.copy()

    # --- 1. Features Agronômicas Básicas ---
    logging.info("Calculando features agronômicas básicas (GDD, VPD)...")
    if 'Tmax (°C)' in df_out.columns and 'Tmin (°C)' in df_out.columns:
        tmean = (df_out['Tmax (°C)'] + df_out['Tmin (°C)']) / 2
        df_out['GDD'] = (tmean - 10).clip(lower=0)  # GDD base 10°C para soja

    if 'Tmed (°C)' in df_out.columns and 'UR (%)' in df_out.columns:
        tmed = df_out['Tmed (°C)']
        ur = df_out['UR (%)']
        es = 0.6108 * np.exp((17.27 * tmed) / (tmed + 237.3))
        df_out['VPD'] = (es * (1 - ur / 100)).clip(lower=0)

    # --- 2. Features de Janela Móvel (Apenas para dados diários/mensais) ---
    if is_daily:
        logging.info("Calculando features de janela móvel (acumulados e estresse)...")
        # Garantir que não há NaNs nas colunas base
        df_out['Chuva (mm)'] = df_out['Chuva (mm)'].fillna(0)
        df_out['GDD'] = df_out['GDD'].fillna(0)
        df_out['Tmax (°C)'] = df_out['Tmax (°C)'].fillna(df_out['Tmed (°C)']) # Fallback

        # Agrupar por município e safra para calcular janelas móveis corretamente
        grouped = df_out.groupby(['municipio', 'SAFRA'])

        for window in [30, 60, 90]:
            df_out[f'Chuva_Acum_{window}d'] = grouped['Chuva (mm)'].transform(
                lambda x: x.rolling(window, min_periods=window//2).sum()
            )
            if 'GDD' in df_out.columns:
                df_out[f'GDD_Acum_{window}d'] = grouped['GDD'].transform(
                    lambda x: x.rolling(window, min_periods=window//2).sum()
                )

        # Features de Estresse
        df_out['Heat_Stress_Flag'] = (df_out['Tmax (°C)'] > 34).astype(int)
        df_out['Heat_Stress_30d'] = grouped['Heat_Stress_Flag'].transform(
            lambda x: x.rolling(30, min_periods=15).sum()
        )
        df_out['Dry_Day_Flag'] = (df_out['Chuva (mm)'] < 1).astype(int)
        df_out['Dry_Days_30d'] = grouped['Dry_Day_Flag'].transform(
            lambda x: x.rolling(30, min_periods=15).sum()
        )

    # --- 3. Features de Sinergia e Polinomiais ---
    logging.info("Calculando features de sinergia e polinomiais...")
    if 'NDVI' in df_out.columns:
        if 'RS (MJ/m²d)' in df_out.columns:
            df_out['NDVI_x_RS'] = df_out['NDVI'] * df_out['RS (MJ/m²d)']
        if 'Chuva_Acum_90d' in df_out.columns:
            df_out['NDVI_x_Chuva_90d'] = df_out['NDVI'] * df_out['Chuva_Acum_90d']
        if 'GDD_Acum_90d' in df_out.columns:
            df_out['NDVI_x_GDD_90d'] = df_out['NDVI'] * df_out['GDD_Acum_90d']
        df_out['NDVI_Quadrado'] = df_out['NDVI'] ** 2

    if 'Tmed (°C)' in df_out.columns:
        df_out['Tmed_Quadrado'] = df_out['Tmed (°C)'] ** 2
        df_out['Desvio_Temp_Otima'] = (df_out['Tmed (°C)'] - 24) ** 2 # Temp ótima ~24°C

    # --- 4. Features de Defasagem (Lag) ---
    logging.info("Calculando features de defasagem (ano anterior)...")
    
    agg_dict_lag = {}
    if 'Chuva (mm)' in df.columns:
        agg_dict_lag['Chuva (mm)'] = 'sum'
    if 'NDVI' in df.columns:
        agg_dict_lag['NDVI'] = 'mean'

    if agg_dict_lag:
        df_safra_avg = df.groupby(['municipio', 'SAFRA']).agg(agg_dict_lag).reset_index()
        
        # Renomear colunas para nomes mais descritivos
        rename_map = {
            'Chuva (mm)': 'Chuva_Total_Safra',
            'NDVI': 'NDVI_Medio_Safra'
        }
        df_safra_avg.rename(columns=rename_map, inplace=True)
        
        df_safra_avg = df_safra_avg.sort_values(by=['municipio', 'SAFRA'])

        # Criar colunas de lag
        lag_cols = list(rename_map.values())
        for col in lag_cols:
            df_safra_avg[f'{col}_Anterior'] = df_safra_avg.groupby('municipio')[col].shift(1)

        # Juntar com o dataframe de saída
        merge_cols = ['municipio', 'SAFRA'] + [f'{c}_Anterior' for c in lag_cols]
        df_out = pd.merge(
            df_out,
            df_safra_avg[merge_cols],
            on=['municipio', 'SAFRA'],
            how='left'
        )

    return df_out


def process_dataset(input_path, output_path):
    """
    Carrega um dataset mestre, aplica a engenharia de features e salva o resultado.
    """
    logging.info(f"Iniciando engenharia de features para: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {input_path}")
        return

    # Ordenar os dados para garantir consistência nos cálculos de janela
    sort_cols = ['municipio', 'SAFRA']
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        sort_cols.append('data')
    elif 'mes' in df.columns:
        sort_cols.append('ano')
        sort_cols.append('mes')
    df.sort_values(by=sort_cols, inplace=True)

    # Determinar se o dataset é diário para aplicar features de janela móvel
    is_daily = 'data' in df.columns

    # Aplicar a função principal de engenharia de features
    df_featured = add_agronomic_features(df, is_daily=is_daily)

    # Salvar o arquivo final
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_featured.to_csv(output_path, index=False)
        logging.info(f"Dataset com features salvo em: {output_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo em {output_path}: {e}")


if __name__ == '__main__':
    # Importar as configurações
    import config

    # Criar o diretório de saída se ele não existir
    os.makedirs(config.FEATURES_DATA_DIR, exist_ok=True)

    # Dicionário com os datasets mestres e seus caminhos de saída
    datasets_to_process = {
        "diario": (config.MASTER_DIARIO_PATH, config.FEATURES_DIARIO_PATH),
        "mensal": (config.MASTER_MENSAL_PATH, config.FEATURES_MENSAL_PATH),
        "anual": (config.MASTER_ANUAL_PATH, config.FEATURES_ANUAL_PATH),
    }

    # Processar cada um dos datasets mestres
    for name, (input_path, output_path) in datasets_to_process.items():
        process_dataset(input_path, output_path)
