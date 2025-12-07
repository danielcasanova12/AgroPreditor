# -*- coding: utf-8 -*-
"""
06_train_and_evaluate.py

Este script é o núcleo de modelagem do projeto. Ele carrega os dados
enriquecidos com features e executa o pipeline de treinamento e avaliação
para os modelos propostos.

Modelos Implementados:
1.  **Modelo Híbrido (LSTM + XGBoost):** Usa uma LSTM para extrair features
    temporais (embeddings) e as combina com features estáticas e agregadas
    para alimentar um modelo XGBoost final.
2.  **Modelo Puro (XGBoost):** Um modelo de baseline que trata os dados de
    forma puramente tabular, usando features de janela móvel para capturar
    a dependência temporal.

O script é parametrizado pelo arquivo `config.py`.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Dropout, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
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

def get_feature_list():
    """Monta a lista de features a partir do config."""
    return (
        config.BASE_FEATURES + config.AGRONOMIC_FEATURES + config.STRESS_FEATURES +
        config.ACCUMULATED_FEATURES + config.INTERACTION_FEATURES +
        config.POLYNOMIAL_FEATURES + config.LAG_FEATURES
    )

def train_hybrid_model(df, granularity):
    """
    Treina e avalia o modelo híbrido LSTM-XGBoost.
    """
    logging.info(f"--- Iniciando Treinamento Híbrido para Granularidade: {granularity} ---")

    # --- 1. Preparação dos Dados ---
    lstm_params = config.LSTM_PARAMS[granularity.upper()]
    max_timesteps = lstm_params['max_timesteps']

    all_features = get_feature_list()
    temporal_features = [f for f in all_features if f in df.columns]
    static_features = [f for f in config.STATIC_FEATURES if f in df.columns]

    logging.info(f"Features temporais utilizadas ({len(temporal_features)}): {temporal_features}")

    # Normalização das features temporais
    scaler = MinMaxScaler()
    df[temporal_features] = scaler.fit_transform(df[temporal_features].fillna(0))

    # --- 2. Construção do Tensor 3D para LSTM ---
    logging.info(f"Construindo tensor 3D com janela de {max_timesteps} timesteps...")
    sequencias, targets, metadata = [], [], []
    
    # Agrupar por safra e município para criar as sequências
    for (muni, safra), group in df.groupby(['municipio', 'SAFRA']):
        # Extrair sequência de features temporais
        seq = group[temporal_features].values
        
        # Padding ou truncamento
        if len(seq) > max_timesteps:
            seq = seq[:max_timesteps]
        else:
            seq = np.pad(seq, ((0, max_timesteps - len(seq)), (0, 0)), mode='constant')
        
        sequencias.append(seq)
        targets.append(group[config.TARGET_COLUMN].iloc[0])
        
        # Coletar metadados (features estáticas e identificadores)
        meta_row = {
            'municipio': muni,
            'SAFRA': safra,
            'is_test': 1 if safra == config.TEST_SAFRA else 0,
            'AREA_TOTAL': group['AREA TOTAL'].iloc[0],
            'PRODUCAO_REAL': group['PRODUCAO'].iloc[0]
        }
        for feat in static_features:
            meta_row[feat] = group[feat].iloc[0]
        metadata.append(meta_row)

    X_seq = np.array(sequencias)
    y = np.array(targets)
    df_meta = pd.DataFrame(metadata)

    # --- 3. Treinamento da LSTM como Extrator de Features ---
    logging.info("Treinando a LSTM para extração de embeddings...")
    
    # Split para treino da LSTM
    train_mask = df_meta['is_test'] == 0
    X_train_nn, y_train_nn = X_seq[train_mask], y[train_mask]

    if len(X_train_nn) == 0:
        logging.error("Não há dados de treino para a LSTM. Abortando.")
        return None

    # Construção do modelo LSTM
    input_layer = Input(shape=(max_timesteps, len(temporal_features)))
    masked = Masking(mask_value=0.0)(input_layer)
    
    if lstm_params['lstm_units'] > 0:
        if lstm_params['bidirectional']:
            x = Bidirectional(LSTM(lstm_params['lstm_units'], return_sequences=False))(masked)
        else:
            x = LSTM(lstm_params['lstm_units'], return_sequences=False)(masked)
        x = Dropout(lstm_params['dropout'])(x)
    else: # Caso ANUAL, usar apenas uma camada Densa
        x = Flatten()(masked)
        x = Dense(64, activation='relu')(x)

    embedding_layer = Dense(config.LSTM_PARAMS['embedding_size'], activation='relu', name='embedding_layer')(x)
    output_layer = Dense(1, activation='linear')(embedding_layer)

    model_nn = Model(inputs=input_layer, outputs=output_layer)
    model_nn.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=config.LSTM_PARAMS['patience'], restore_best_weights=True)
    model_nn.fit(X_train_nn, y_train_nn, epochs=config.LSTM_PARAMS['epochs'], batch_size=config.LSTM_PARAMS['batch_size'], verbose=0, callbacks=[es])

    # --- 4. Extração de Embeddings e Construção do Dataset Final para XGBoost ---
    logging.info("Extraindo embeddings e construindo dataset para o XGBoost.")
    
    # Usar a LSTM treinada para extrair os embeddings de todos os dados (treino e teste)
    extractor = Model(inputs=model_nn.input, outputs=model_nn.get_layer('embedding_layer').output)
    embeddings = extractor.predict(X_seq)
    df_emb = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(config.LSTM_PARAMS['embedding_size'])])

    # Juntar metadados, features estáticas e embeddings
    df_xgb = pd.concat([df_meta.reset_index(drop=True), df_emb.reset_index(drop=True)], axis=1)

    # Adicionar a coluna alvo ao dataframe antes de fazer o split
    df_xgb[config.TARGET_COLUMN] = y

    # Codificar features categóricas
    for col in config.STATIC_FEATURES:
        if col in df_xgb.columns:
            df_xgb[col] = df_xgb[col].astype('category').cat.codes

    # --- 5. Treinamento e Avaliação do XGBoost ---
    logging.info("Treinando e avaliando o modelo XGBoost final.")
    
    # Split final
    df_train = df_xgb[df_xgb['is_test'] == 0]
    df_test = df_xgb[df_xgb['is_test'] == 1]

    if df_train.empty or df_test.empty:
        logging.warning("Dados de treino ou teste para o XGBoost estão vazios.")
        return None

    xgb_features = [f'emb_{i}' for i in range(config.LSTM_PARAMS['embedding_size'])] + [col for col in config.STATIC_FEATURES if col in df_xgb.columns]
    
    X_train = df_train[xgb_features]
    y_train = df_train[config.TARGET_COLUMN]
    X_test = df_test[xgb_features]
    y_test = df_test[config.TARGET_COLUMN]

    # Tuning e treinamento do XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=config.XGB_TUNING_PARAMS,
        n_iter=config.N_ITER_SEARCH,
        scoring='neg_mean_squared_error',
        cv=config.CV_FOLDS,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    logging.info(f"Melhores parâmetros para o XGBoost: {random_search.best_params_}")

    # Avaliação
    y_pred = best_xgb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logging.info(f"Resultados para {granularity} (Híbrido): R² = {r2:.4f}, RMSE = {rmse:.4f} sc/ha")

    # TODO: Salvar modelo e resultados
    
    return {'r2': r2, 'rmse': rmse, 'model': best_xgb, 'features': xgb_features}


if __name__ == '__main__':
    # Criar diretórios de resultados se não existirem
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Mapeamento de granularidade para caminhos de arquivo
    feature_paths = {
        "diario": config.FEATURES_DIARIO_PATH,
        "mensal": config.FEATURES_MENSAL_PATH,
        "anual": config.FEATURES_ANUAL_PATH,
    }

    results = {}
    for granularity, path in feature_paths.items():
        try:
            df = pd.read_csv(path)
            # A linha abaixo era redundante e foi removida.
            # O arquivo de features já contém a coluna alvo.
            df.dropna(subset=[config.TARGET_COLUMN], inplace=True)

            # Executar o pipeline de modelo híbrido
            result = train_hybrid_model(df, granularity)
            if result:
                results[f'hibrido_{granularity}'] = result

            # TODO: Adicionar chamada para um modelo XGBoost puro como baseline

        except FileNotFoundError:
            logging.error(f"Arquivo de features não encontrado para {granularity}: {path}")
        except Exception as e:
            logging.error(f"Ocorreu um erro ao processar a granularidade {granularity}: {e}")

    logging.info("\n--- RESUMO FINAL DOS RESULTADOS ---")
    for model_name, metrics in results.items():
        logging.info(f"{model_name: <20} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} sc/ha")
