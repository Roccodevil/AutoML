---
title: AutoML
emoji: 🏃
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: An Agentic AI workflow to counter every ML/Data problem
---

# 🏃 Agentic AI AutoML Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![AutoML Engine](https://img.shields.io/badge/AutoML-H2O%20%7C%20Optuna-green.svg)](https://h2o.ai)
[![UI](https://img.shields.io/badge/UI-Flask%20%7C%20Bootstrap-blueviolet.svg)](https://flask.palletsprojects.com/)
[![Docker Container](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An autonomous, multi-agent AI system designed to solve end-to-end Machine Learning, Data Science, and Natural Language Processing (NLP) challenges. Powered by **LangGraph**, **H2O AutoML**, **Optuna**, and **Hugging Face Transformers**, this platform automates the complete data lifecycle—from ingestion and exploratory data analysis to feature engineering, model optimization, and deployment packaging.

---

## 🌟 Key Features

- 🧠 **Multi-Agent Orchestration**: Modular, autonomous agents coordinated using LangGraph for multi-stage workflow execution.
- 📥 **Multi-Source Data Ingestion**:
  - File upload support: CSV, Excel (`.xlsx`), PDF, Word (`.docx`), and Plain Text.
  - Integration with **Hugging Face Hub** and **Kaggle API** for automated online dataset search.
  - Synthetic data generation powered by LLMs (Groq / LangChain) and Tavily web search context.
- 📊 **Automated Exploratory Data Analysis (EDA)**: Statistical profiling, missingness diagnosis, distribution analysis, and data health assessments.
- 🛠️ **Smart Preprocessing & Feature Engineering**: Automated categorical encoding, missing value imputation, outlier handling, interaction terms creation, and mathematical transformations.
- 📈 **Automated Visualization**: Publication-ready charts including histograms, correlation heatmaps, target relationships, and feature distributions.
- 🚀 **Dual Pipeline Engines**:
  - **Tabular Data AutoML**: Powered by H2O AutoML and Optuna hyperparameter optimization.
  - **AutoNLP Pipeline**: Task identification, text cleaning, feature extraction, and fine-tuning with Hugging Face Transformers and PyTorch.
- 🖥️ **Interactive Node-Based Control Center**: Visual graph editor for customized pipeline construction, real-time log streaming, and MongoDB execution tracking.
- 📦 **One-Click Export & Deployment**: Bundles trained model artifacts, diagnostic charts, text summary reports, and a ready-to-deploy Flask/Gunicorn web app into executable zip packages.

---

## 🏗️ Architecture & Pipeline Agents

The platform executes pipelines through specialized autonomous agents:

```
                  ┌─────────────────────────────────┐
                  │   Agent 1: Data Ingestion       │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 2: Data Analysis & EDA  │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 3: Preprocessing        │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 4: Visualization        │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 5: Feature Engineering  │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 6: Data Staging         │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 7: AutoML & Tuning      │
                  │   (H2O / Optuna)                │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   Agent 8: Export & Deployment  │
                  └─────────────────────────────────┘
```

### Agent Roles:
1. **Agent 1 (Data Ingestion)**: Acquires data via upload, Kaggle, Hugging Face, or LLM-guided synthetic generation.
2. **Agent 2 (Data Analysis)**: Computes statistical profiles, checks missing values, and determines target variable types.
3. **Agent 3 (Preprocessing)**: Imputes missing values, encodes categorical variables, scales numeric features, and handles anomalies.
4. **Agent 4 (Visualization)**: Generates and saves analytical figures and correlation charts.
5. **Agent 5 (Feature Engineering)**: Synthesizes high-impact interaction features and performs automated feature selection.
6. **Agent 6 (Staging)**: Splits data into train/test sets with stratified sampling.
7. **Agent 7 (AutoML)**: Executes parallel model training (GBM, Random Forest, XGBoost, Deep Learning via H2O / Optuna) and ranks models by performance metrics.
8. **Agent 8 (Export)**: Packages production model artifacts, diagnostic visualizations, and deployment scripts into zipped archives.

---

## 🧰 Tech Stack

- **Backend & Web Framework**: Flask, Gunicorn, MongoDB (`flask-pymongo`)
- **Pipeline & LLM Orchestration**: LangGraph, LangChain, `langchain-groq`, `tavily-python`
- **AutoML & Machine Learning**: H2O AutoML, Optuna, Scikit-Learn, Pandas, NumPy, Statsmodels
- **NLP & Deep Learning**: PyTorch (CPU optimized), Hugging Face `transformers`, NLTK, spaCy, TextBlob
- **Data Ingestion**: `datasets`, `huggingface_hub`, `kaggle`, `pdfplumber`, `python-docx`, `openpyxl`
- **Visualization**: Matplotlib, Seaborn, Pillow
- **Deployment & Containerization**: Docker, Java Runtime (JRE for H2O), Render / Hugging Face Spaces compatibility

---

## ⚙️ Installation & Setup

### Prerequisites

- Python **3.10** installed
- Java Runtime Environment (JRE) required for H2O AutoML
- MongoDB server (local instance or MongoDB Atlas cluster)

### 1. Clone the Repository

```bash
git clone https://github.com/Roccodevil/AutoML.git
cd AutoML
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
MONGO_URI=mongodb://localhost:27017/automl
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## 🚀 Running the Application

### Local Flask Development Server

Run the application using Python:

```bash
python run.py
```

The web dashboard will be available at `http://localhost:5000`.

### Docker Deployment

To build and run the containerized application:

```bash
# Build Docker Image
docker build -t automl-platform .

# Run Container
docker run -p 10000:10000 --env-file .env automl-platform
```

Access the service at `http://localhost:10000`.

---

## 📖 Usage Guide

1. **Dashboard Overview**: Access `http://localhost:5000/` for general management or `http://localhost:5000/automl` for tabular pipelines.
2. **AutoNLP Workspace**: Access `http://localhost:5000/autonlp` to execute NLP workloads.
3. **Data Acquisition**: Select local file upload, Hugging Face search, Kaggle search, or text description generation.
4. **Pipeline Execution**: View real-time node execution status and logs.
5. **Download Artifacts**: Retrieve trained model packages, chart bundles, evaluation reports, and deployable web server zip archives.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
