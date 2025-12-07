# Previsão de Produtividade da Soja com Modelos Híbridos
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Projeto de dissertação de mestrado que explora o uso de um modelo híbrido (LSTM + XGBoost) para prever a produtividade da soja (sc/ha) em municípios do estado do Paraná, utilizando dados climáticos, de produção e de sensoriamento remoto (NDVI).

---

## 📜 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
  - [Execução do Pipeline](#execução-do-pipeline)
- [Resultados](#-resultados)
- [Metodologia](#-metodologia)
- [Trabalhos Futuros](#-trabalhos-futuros)
- [Autor](#-autor)

---

## 📖 Sobre o Projeto

Este trabalho investiga a viabilidade de prever a produtividade da soja em nível municipal com base em dados públicos. A hipótese central é que a combinação de uma Rede Neural Recorrente (LSTM) para interpretar a dinâmica temporal dos dados climáticos com um modelo de Gradient Boosting (XGBoost) para integrar características estáticas e agronômicas pode gerar previsões robustas.

**Fontes de Dados:**
- **Produtividade:** Sistema IBGE de Recuperação Automática (SIDRA).
- **Clima:** Dados diários de estações meteorológicas governamentais (provavelmente SIMEPAR/TECPAR).
- **NDVI:** Imagens de satélite (fonte específica não documentada).

**Modelo:** Híbrido LSTM-XGBoost.

---

## 📂 Estrutura do Projeto

O projeto foi organizado em um pipeline de scripts sequenciais para garantir a clareza e reprodutibilidade do processo.

```
.
├── 📄 config.py                   # Arquivo central de configuração
├── 📄 01_prepare_ndvi.py          # Limpa e filtra dados de NDVI
├── 📄 02_prepare_yield.py         # Calcula a produtividade (alvo)
├── 📄 03_prepare_climate.py       # Filtra dados climáticos para a safra
├── 📄 04_create_master_datasets.py # Unifica os dados em datasets mestre
├── 📄 05_feature_engineering.py   # Cria todos os atributos agronômicos
├── 📄 06_train_and_evaluate.py    # Treina e avalia os modelos
├── 📄 07_generate_report.py       # Gera o relatório de metodologia
├── 📄 08_plot_correlation_matrix.py # Gera a matriz de correlação
├── 📄 requirements.txt            # Lista de dependências do Python
├── 📁 data/                       # Contém os dados brutos e processados
└── 📁 results/                    # Contém os outputs (gráficos, relatórios)
```

---

## 🚀 Como Executar

Siga os passos abaixo para configurar o ambiente e executar o pipeline completo.

### Pré-requisitos

- [Python 3.11+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads/)

### Instalação

1.  **Clone o repositório:**
    ```sh
    git clone https://github.com/danielcasanova12/AgroPreditor.git
    cd AgroPreditor
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    O arquivo `requirements.txt` contém todas as bibliotecas necessárias.
    ```sh
    pip install -r requirements.txt
    ```

### Execução do Pipeline

Os scripts devem ser executados na ordem numérica para garantir que os dados sejam processados corretamente.

```sh
python 01_prepare_ndvi.py
python 02_prepare_yield.py
python 03_prepare_climate.py
python 04_create_master_datasets.py
python 05_feature_engineering.py
python 06_train_and_evaluate.py
python 07_generate_report.py
python 08_plot_correlation_matrix.py
```

Ao final, os resultados, o relatório e os gráficos estarão na pasta `results/`.

---

## 📊 Resultados

A execução do pipeline com o modelo híbrido na safra de teste ('23/24') produziu os seguintes resultados:

| Granularidade | R² (Coef. de Determinação) | RMSE (sacas/ha) |
|---------------|------------------------------|-------------------|
| **Mensal**    | **0.8132**                   | **7655.61**       |
| Anual         | 0.6585                       | 9122.39           |
| Diário        | 0.6641                       | 9325.73           |

O modelo com **dados mensais** apresentou o melhor desempenho. A matriz de correlação abaixo explora a relação entre as principais variáveis neste dataset.

![Matriz de Correlação](results/correlation_matrix_mensal.png)

---

## 📝 Metodologia

Uma descrição detalhada de toda a metodologia, incluindo as fontes de dados, o pré-processamento, a engenharia de atributos e a arquitetura do modelo, está disponível no arquivo:
[**results/metodologia_e_discussao.txt**](results/metodologia_e_discussao.txt)

---

## 🔮 Trabalhos Futuros

- **Validação Robusta:** Implementar uma Validação Cruzada Deixando Um Ano de Fora (Leave-One-Year-Out CV) para obter uma métrica de performance mais estável.
- **Features Fenológicas:** Alinhar a agregação de features com os estádios fenológicos da soja (Vn, Rn) em vez de janelas de calendário fixas.
- **Dados de Solo:** Enriquecer o modelo com dados quantitativos do solo (ex: % de argila, pH).
- **Análise SHAP:** Aprofundar a análise de interpretabilidade do modelo final para validar as hipóteses agronômicas.

---

