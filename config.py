# -*- coding: utf-8 -*-
"""
config.py

Arquivo de configuração central para o projeto de previsão de safra.
Este arquivo armazena todos os caminhos, listas, parâmetros e constantes
para facilitar a manutenção e a experimentação.
"""
import os

# --- 1. CAMINHOS (PATHS) ---
# Diretório base para os dados
BASE_DATA_DIR = './data'

# Subdiretórios para cada etapa do pipeline
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, 'raw') # Recomendado mover os arquivos originais para cá
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'processed')
MASTER_DATA_DIR = os.path.join(BASE_DATA_DIR, 'master')
FEATURES_DATA_DIR = os.path.join(BASE_DATA_DIR, 'features')
RESULTS_DIR = './results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# Caminhos para arquivos de entrada (assumindo que estão no diretório base)
# TODO: Mover arquivos originais para RAW_DATA_DIR
NDVI_RAW_PATH = os.path.join(BASE_DATA_DIR, 'NDVI_Municipios_unico.csv')
YIELD_RAW_PATH = os.path.join(BASE_DATA_DIR, 'soja_por_ano_municipio_area.csv')
CLIMATE_RAW_PATH = os.path.join(BASE_DATA_DIR, 'clima_PR_2000-2024_clean.csv')
CLIMATE_FULL_RAW_PATH = os.path.join(BASE_DATA_DIR, 'dados_climaticos_parana_completo.csv')

# Caminhos para arquivos processados (Etapas 01-03)
NDVI_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'ndvi_filtrado.csv')
YIELD_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'yield_calculado.csv')
CLIMATE_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'clima_safra.csv')

# Caminhos para datasets mestres (Etapa 04)
MASTER_DIARIO_PATH = os.path.join(MASTER_DATA_DIR, 'master_diario.csv')
MASTER_MENSAL_PATH = os.path.join(MASTER_DATA_DIR, 'master_mensal.csv')
MASTER_ANUAL_PATH = os.path.join(MASTER_DATA_DIR, 'master_anual.csv')

# Caminhos para datasets com features (Etapa 05)
FEATURES_DIARIO_PATH = os.path.join(FEATURES_DATA_DIR, 'features_diario.csv')
FEATURES_MENSAL_PATH = os.path.join(FEATURES_DATA_DIR, 'features_mensal.csv')
FEATURES_ANUAL_PATH = os.path.join(FEATURES_DATA_DIR, 'features_anual.csv')


# --- 2. PARÂMETROS GERAIS ---
# Lista de municípios de interesse para a análise
MUNICIPIOS_DE_INTERESSE = [
    "Lindoeste", "Bandeirantes", "Imbaú", "Antonina", "Pinhal de São Bento",
    "Nova Esperança do Sudoeste", "Campina do Simão", "Diamante do Norte",
    "Cruzeiro do Sul", "Wenceslau Braz", "Francisco Alves", "Moreira Sales",
    "Mato Rico", "Diamante do Sul",
]

# Meses que compõem a safra da soja
MESES_DA_SAFRA = [9, 10, 11, 12, 1, 2, 3]

# Safra a ser utilizada como conjunto de teste
TEST_SAFRA = '23/24'


# --- 3. PARÂMETROS DE MODELAGEM ---
# Coluna alvo
TARGET_COLUMN = 'YIELD_SC_HA'

# Features a serem usadas pelo modelo.
# Deixar a lista explícita aqui ajuda na interpretabilidade e controle.
# Esta lista será preenchida com as features geradas na etapa 05.
BASE_FEATURES = [
    'Tmax (°C)', 'Tmin (°C)', 'Tmed (°C)', 'UR (%)', 'U2 (m/s)',
    'RS (MJ/m²d)', 'Chuva (mm)', 'NDVI'
]

AGRONOMIC_FEATURES = [
    'GDD', 'VPD'
]

STRESS_FEATURES = [
    'Heat_Stress_30d', 'Dry_Days_30d'
]

ACCUMULATED_FEATURES = [
    'Chuva_Acum_30d', 'Chuva_Acum_60d', 'Chuva_Acum_90d',
    'GDD_Acum_30d', 'GDD_Acum_60d', 'GDD_Acum_90d'
]

INTERACTION_FEATURES = [
    'NDVI_x_RS', 'NDVI_x_Chuva_90d', 'NDVI_x_GDD_90d'
]

POLYNOMIAL_FEATURES = [
    'NDVI_Quadrado', 'Tmed_Quadrado', 'Desvio_Temp_Otima'
]

LAG_FEATURES = [
    'Chuva_Media_Safra_Anterior', 'NDVI_Media_Safra_Anterior'
]

STATIC_FEATURES = ['Solo', 'REGIAO', 'municipio'] # Features categóricas

# Parâmetros para o modelo Híbrido (LSTM)
LSTM_PARAMS = {
    'DIARIO': {
        'max_timesteps': 210, # Aprox. 7 meses
        'lstm_units': 64,
        'dropout': 0.2,
        'bidirectional': False
    },
    'MENSAL': {
        'max_timesteps': 7, # 7 meses
        'lstm_units': 32,
        'dropout': 0.1,
        'bidirectional': True
    },
    'ANUAL': {
        'max_timesteps': 1, # Apenas 1 entrada
        'lstm_units': 0, # Não usa LSTM
    },
    'embedding_size': 32,
    'epochs': 100,
    'batch_size': 16,
    'patience': 15
}

# Espaço de busca para o RandomizedSearchCV do XGBoost
XGB_TUNING_PARAMS = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Número de iterações para o RandomizedSearchCV
N_ITER_SEARCH = 25
CV_FOLDS = 5 # Folds para a validação cruzada do tuning

# Parâmetros para o Random Forest do ensemble
RF_ENSEMBLE_PARAMS = {
    'n_estimators': 300,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}
