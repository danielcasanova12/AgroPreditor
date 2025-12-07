# -*- coding: utf-8 -*-
"""
08_plot_correlation_matrix.py

Este script gera e salva uma matriz de correlação para visualizar a relação
entre as features mais importantes e a variável alvo.

Utiliza o dataset mensal, que apresentou os melhores resultados de modelo.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

# Importar configurações
import config

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution.log", mode='a'),
        logging.StreamHandler()
    ]
)

def plot_correlation_matrix(data_path, output_path):
    """
    Carrega os dados, calcula e plota a matriz de correlação.
    """
    logging.info(f"Gerando matriz de correlação para o arquivo: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Arquivo de dados não encontrado: {data_path}")
        return

    # Selecionar um subconjunto de features para manter a matriz legível
    # Incluímos o alvo, features agronômicas, acumuladas e de lag.
    feature_subset = [config.TARGET_COLUMN] + \
                     config.BASE_FEATURES + \
                     config.AGRONOMIC_FEATURES + \
                     config.LAG_FEATURES
    
    # Garantir que apenas colunas existentes no dataframe sejam usadas
    feature_subset = [col for col in feature_subset if col in df.columns]
    
    df_subset = df[feature_subset]

    # Renomear colunas para melhor visualização no gráfico
    df_subset.rename(columns={
        'YIELD_SC_HA': 'Produtividade (sc/ha)',
        'Tmax (°C)': 'Temp. Max (C)',
        'Tmin (°C)': 'Temp. Min (C)',
        'Tmed (°C)': 'Temp. Med (C)',
        'UR (%)': 'Umidade Rel. (%)',
        'RS (MJ/m²d)': 'Rad. Solar',
        'Chuva (mm)': 'Chuva (mm)',
        'Chuva_Total_Safra_Anterior': 'Chuva Safra Ant.',
        'NDVI_Medio_Safra_Anterior': 'NDVI Safra Ant.'
    }, inplace=True)

    # Calcular a matriz de correlação
    corr_matrix = df_subset.corr()

    # Gerar o gráfico (heatmap)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=.5,
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    ax.set_title('Matriz de Correlação - Features Mensais', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Salvar a imagem
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logging.info(f"Matriz de correlação salva em: {output_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar o gráfico: {e}")


if __name__ == '__main__':
    # Caminho para o dataset com features e para o arquivo de saída do gráfico
    DATA_FILE_PATH = config.FEATURES_MENSAL_PATH
    OUTPUT_PLOT_PATH = os.path.join(config.RESULTS_DIR, 'correlation_matrix_mensal.png')

    plot_correlation_matrix(DATA_FILE_PATH, OUTPUT_PLOT_PATH)
