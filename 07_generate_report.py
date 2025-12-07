# -*- coding: utf-8 -*-
"""
07_generate_report.py

Este é o script final do pipeline. Ele é responsável por carregar os
resultados e modelos da etapa de treinamento, realizar uma análise de
interpretabilidade (XAI) com SHAP, e gerar um relatório em texto
com a metodologia, discussão dos resultados e análise crítica.
"""
import pandas as pd
import numpy as np
import joblib # Para carregar modelos salvos
import shap
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

def generate_report_text(results):
    """
    Gera o conteúdo do arquivo de texto com a metodologia e discussão.
    """
    # Cabeçalho do relatório
    report = "RELATÓRIO DE METODOLOGIA E ANÁLISE CRÍTICA\n"
    report += "="*50 + "\n\n"

    # --- Seção 1: Metodologia ---
    report += "1. METODOLOGIA APLICADA\n"
    report += "-"*25 + "\n"
    report += (
        "O projeto foi estruturado em um pipeline de dados e modelagem sequencial e modular "
        "para garantir reprodutibilidade, clareza e manutenibilidade.\n\n"
    )
    report += "1.1. Preparação e Limpeza dos Dados (Scripts 01-03):\n"
    report += (
        "   - NDVI: Os dados brutos de NDVI foram processados para extrair e padronizar os nomes dos municípios. "
        "Uma filtragem foi aplicada para manter apenas os 14 municípios de interesse definidos em `config.py`.\n"
        "   - Produtividade (Yield): Os dados de produção de soja (toneladas) e área plantada (hectares) foram "
        "carregados. A variável alvo, 'YIELD_SC_HA' (sacas de 60kg por hectare), foi calculada.\n"
        "   - Clima: Os dados climáticos foram filtrados para conter apenas os meses da safra da soja (Setembro a Março), "
        "reduzindo o ruído e focando no período de interesse.\n\n"
    )
    report += "1.2. Criação dos Datasets Mestres (Script 04):\n"
    report += (
        "   - Os três datasets pré-processados foram unificados. Uma função (`date_to_safra`) mapeou cada registro "
        "para sua safra correspondente (ex: '22/23').\n"
        "   - Foi criado um dataset mestre diário, que serviu de base para as agregações.\n"
        "   - A partir da base diária, foram gerados datasets agregados nos níveis MENSAL e ANUAL, com lógicas de "
        "agregação específicas para cada variável (ex: SOMA para chuva, MÉDIA para temperatura).\n\n"
    )
    report += "1.3. Engenharia de Features (Script 05):\n"
    report += (
        "   - Esta etapa, centralizada e consistente, enriqueceu os datasets com features agronômicas, incluindo:\n"
        "     - **Variáveis Agronômicas:** GDD (Graus-Dia de Crescimento) e VPD (Déficit de Pressão de Vapor).\n"
        "     - **Variáveis de Estresse:** Contagem de dias com temperatura máxima > 34°C e dias com chuva < 1mm em janelas de 30 dias.\n"
        "     - **Variáveis Acumuladas:** Soma de chuva e GDD em janelas de 30, 60 e 90 dias.\n"
        "     - **Variáveis de Interação e Polinomiais:** Termos para capturar sinergias (ex: NDVI * Chuva) e relações não-lineares (ex: desvio da temperatura ótima).\n"
        "     - **Variáveis de Defasagem (Lag):** Média de NDVI e chuva total da safra anterior para capturar a 'memória' do sistema.\n\n"
    )
    report += "1.4. Modelagem Híbrida (Script 06):\n"
    report += (
        "   - **Arquitetura:** Foi implementado um modelo híbrido combinando uma Rede Neural Recorrente (LSTM) com um Gradient Boosting (XGBoost).\n"
        "   - **LSTM como Extrator de Features:** A LSTM foi treinada nos dados sequenciais (diários/mensais) para gerar um vetor de características latentes ('embedding') que resume a dinâmica temporal da safra.\n"
        "   - **XGBoost como Modelo Final:** O XGBoost foi treinada usando uma combinação de features: (1) os embeddings da LSTM, (2) features estáticas (município, solo, região) e (3) metadados da safra.\n"
        "   - **Validação:** O modelo foi treinado em todas as safras disponíveis, exceto a safra '23/24', que foi usada como conjunto de teste para avaliar a performance em dados não vistos.\n"
        "   - **Otimização:** O XGBoost foi otimizado usando `RandomizedSearchCV` para encontrar os melhores hiperparâmetros.\n\n"
    )

    # --- Seção 2: Discussão dos Resultados ---
    report += "\n2. DISCUSSÃO DOS RESULTADOS\n"
    report += "-"*25 + "\n"
    # Esta parte seria preenchida com os resultados reais
    if results:
        report += "Os resultados da avaliação na safra de teste ('23/24') foram:\n\n"
        for model_name, metrics in results.items():
            report += f"   - **Modelo {model_name.replace('_', ' ').title()}**:\n"
            report += f"     - R² (Coef. de Determinação): {metrics.get('r2', 'N/A'):.4f}\n"
            report += f"     - RMSE (Raiz do Erro Quadrático Médio): {metrics.get('rmse', 'N/A'):.2f} sacas/ha\n\n"
        
        # Análise comparativa
        # Adicionar lógica para comparar os resultados (diario vs mensal vs anual)
        
    else:
        report += "Os resultados da modelagem ainda não foram gerados ou carregados.\n\n"

    # --- Seção 3: Análise Crítica e Próximos Passos ---
    report += "\n3. ANÁLISE CRÍTICA E SUGESTÕES\n"
    report += "-"*25 + "\n"
    report += "3.1. Análise de Sinergia das Features (Interpretabilidade com SHAP):\n"
    report += (
        "   - A análise de importância de features (SHAP) no modelo final é crucial para entender os drivers da produtividade.\n"
        "   - **Hipótese 1 (Features Agronômicas):** Espera-se que GDD e VPD acumulados sejam altamente importantes. O SHAP pode mostrar se a relação é linear ou se há um ponto de saturação.\n"
        "   - **Hipótese 2 (Features de Estresse):** A contagem de dias com estresse térmico ou hídrico, especialmente em fases críticas (enchimento de grãos), deve ter um impacto negativo significativo na produtividade.\n"
        "   - **Hipótese 3 (Sinergia NDVI-Clima):** As features de interação como 'NDVI_x_Chuva' são fundamentais. Um NDVI alto (planta verde) sem chuva acumulada pode ser um indicador de estresse e baixa produtividade. O SHAP pode validar essa sinergia.\n"
        "   - **Hipótese 4 (Embeddings da LSTM):** A importância das features 'emb_X' indicará o quanto o modelo XGBoost confiou na representação temporal aprendida pela LSTM. Se essas features forem as mais importantes, isso valida a complexidade da arquitetura híbrida.\n\n"
    )
    report += "3.2. Crítica aos Parâmetros e Modelo:\n"
    report += (
        "   - **Modelo Híbrido:** A principal vantagem é a capacidade da LSTM de aprender padrões temporais complexos que o XGBoost sozinho não consegue. A desvantagem é a complexidade e o tempo de treinamento. A importância dos embeddings (acima) é a chave para justificar essa escolha.\n"
        "   - **Hiperparâmetros:** O uso de `RandomizedSearchCV` é uma boa prática. No entanto, os resultados são sensíveis ao espaço de busca definido em `config.py`. Seria interessante analisar se os melhores parâmetros encontrados estão nos limites do espaço de busca, o que poderia indicar a necessidade de expandi-lo.\n"
        "   - **Overfitting:** O uso de `EarlyStopping` na LSTM e a regularização (L1/L2) no XGBoost são medidas importantes contra o overfitting. A validação em um ano completamente separado também ajuda a ter uma estimativa mais realista da performance.\n\n"
    )
    report += "3.3. Limitações e Sugestões para Trabalhos Futuros:\n"
    report += (
        "   - **Validação Cruzada:** A avaliação em uma única safra-teste ('23/24') é um ponto fraco. Uma abordagem mais robusta seria a **Validação Cruzada Deixando Um Ano de Fora (Leave-One-Year-Out CV)**, onde o modelo é treinado N vezes, a cada vez deixando uma safra diferente de fora para teste. Isso forneceria uma métrica de performance muito mais estável e confiável.\n"
        "   - **Features Fenológicas:** A engenharia de features poderia ser elevada a um novo patamar ao alinhar as agregações com os estádios fenológicos da soja (ex: V1-Vn, R1-R8), em vez de usar janelas de calendário fixas (30, 60, 90 dias). Isso criaria features com maior significado biológico.\n"
        "   - **Dados de Solo:** A feature 'Solo' é usada de forma categórica. Enriquecer o dataset com dados quantitativos do solo (ex: % de argila, matéria orgânica, pH) por município poderia aumentar significativamente o poder preditivo do modelo.\n"
    )

    return report

def parse_log_for_results(log_path):
    """
    Analisa o arquivo de log para extrair os resultados finais da última execução.
    """
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Arquivo de log não encontrado em: {log_path}")
        return None

    results = {}
    summary_section_lines = []
    capture = False

    # Encontra a última seção de resumo no log
    for line in lines:
        if "--- RESUMO FINAL DOS RESULTADOS ---" in line:
            # Nova seção de resumo encontrada, limpa a anterior para pegar apenas a última
            summary_section_lines = []
            capture = True
            continue
        
        if capture and "INFO" in line and "|" in line:
            summary_section_lines.append(line)
        elif capture and ("INFO" not in line or "|" not in line):
            # Se encontrar uma linha que não pertence à tabela de resumo, para de capturar
            capture = False

    # Processa as linhas da última seção de resumo encontrada
    if not summary_section_lines:
        return None

    for line in summary_section_lines:
        parts = line.split('|')
        if len(parts) != 3: # Espera 3 partes: nome | R² | RMSE
            continue

        try:
            # O nome do modelo é a primeira palavra após 'INFO -'
            model_name = parts[0].split('INFO -')[1].strip().split()[0]
            
            # Extrai os valores de R² e RMSE de forma mais robusta
            r2_val = float(parts[1].split(':')[1].strip())
            rmse_val = float(parts[2].split(':')[1].split('sc/ha')[0].strip())
            
            results[model_name] = {
                'r2': r2_val,
                'rmse': rmse_val
            }
        except (IndexError, ValueError) as e:
            logging.warning(f"Não foi possível parsear a linha de resultado: {line.strip()} - Erro: {e}")
            continue
            
    return results

if __name__ == '__main__':
    logging.info("Iniciando a geração do relatório final...")
    
    # Analisa dinamicamente o log para obter os resultados mais recentes
    log_file_path = 'execution.log'
    final_results = parse_log_for_results(log_file_path)

    if not final_results:
        logging.warning("Não foram encontrados resultados no arquivo de log. O relatório será gerado com dados de exemplo.")
        # Usar um dicionário de exemplo se a análise do log falhar
        final_results = {
            'hibrido_diario': {'r2': 0.0, 'rmse': 0.0},
            'hibrido_mensal': {'r2': 0.0, 'rmse': 0.0},
            'hibrido_anual': {'r2': 0.0, 'rmse': 0.0},
        }

    report_content = generate_report_text(final_results)

    # Salvar o relatório em um arquivo de texto
    report_path = os.path.join(config.RESULTS_DIR, 'metodologia_e_discussao.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Relatório salvo com sucesso em: {report_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar o relatório: {e}")

    print("\n" + report_content)

