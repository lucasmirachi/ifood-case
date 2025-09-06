# Projeto Data Science - Case Técnico Ifood
---

<img src="capa.png" width="900px">

## Estrutura do repositório

```
ifood-case
├── data/
│   ├── raw/                    # Dados originais
│   └── processed/              # Dados processados
├── notebooks/
│   ├── 1_data_processing.ipynb # Processamento de dados
│   └── 2_modeling.ipynb        # Modelagem ML
├── presentation/               # Apresentações
└── requirements.txt
```

## Setup no Databricks Community

### 1. Upload dos Notebooks
1. Acessar [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. Ir em **Workspace** > **Users** > **Seu email**
3. Clicar em  **Import** e fazer upload dos arquivos `.ipynb`

### 2. Configurar Cluster
1. Vá em **Compute** > **Create Cluster** (ou **Start** se existir)
2. Aguarde ficar **Running** (~3 minutos)

### 3. Upload de Dados
1. Vá em **Data** > **Create Table**
2. Faça upload dos arquivos (máximo 15GB total)
3. Dados ficarão em `/FileStore/tables/`

## Execução

### Ordem:
1. Execute `1_data_processing.ipynb`
2. Execute `2_modeling.ipynb`

### Instalar bibliotecas (início de cada notebook):

Na versão Databricks Community, todas as libs utilizadas na solução já estão instaladas, mas caso seja necessário, executar:

```python
%pip install pandas scikit-learn matplotlib seaborn
dbutils.library.restartPython()
```

### Configurar paths:
```python
DATA_PATH = "/dbfs/FileStore/tables/"
```

```