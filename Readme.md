# AI-Driven Forecasting and Waste Optimization for Perishable Retail Goods in Ecuador
### Part 1 — Data Engineering & Forecasting Pipeline

> This repository is **Part 1** of a two-part project.
> The agentic AI and demand intelligence module can be found here → [perishable-retail-agent](YOUR_LINK_HERE)

---

## Project Overview
This capstone project builds a demand forecasting system for perishable retail goods using the Favorita grocery sales dataset from Ecuador. The goal is to combine historical sales with external signals such as weather, holidays, oil prices, and store-level transaction activity to improve forecast accuracy and support waste-aware inventory planning.

## Current Scope
This repository currently includes the work completed for **Status Update 1**:
- Data ingestion into PostgreSQL
- Spark-based ETL and integration
- Creation of a star schema
- External feature integration
- Initial feature engineering
- Baseline and XGBoost forecasting models
- Exploratory data analysis

## Data Sources
### Main Dataset
- **Favorita Grocery Sales Dataset**
  - Daily store-item sales
  - Store metadata
  - Item metadata
  - Transactions
  - Oil prices
  - Holiday/event information

### External Data
- City-level weather data for Ecuador
- FAOSTAT production data
- FAOSTAT import/export data

## Project Goals
- Forecast daily perishable product demand
- Evaluate how external factors affect sales
- Compare baseline and ML model performance
- Quantify predictive improvement using MAE and related metrics

## Current Workflow
1. Raw Favorita data loaded into PostgreSQL
2. External datasets cleaned and stored in feature tables
3. Spark used to join core datasets into a modeling base
4. Feature engineering performed in Python/Colab
5. Baseline and XGBoost models trained and evaluated
6. Initial visualizations prepared for Status Update 1

## Key Outputs So Far
- Star schema warehouse in PostgreSQL
- Modeling base parquet export
- Lag-based and rolling features
- Baseline MAE
- XGBoost MAE
- Feature importance analysis

## Data
The raw dataset used in this project is publicly available on Kaggle:
[Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

Download and place files under `Data/Raw/` before running the pipeline.

## Repository Structure
```text
perishable-retail-forecasting-pipeline/
├── Data/
│   ├── Raw/          ← download from Kaggle (not tracked in git)
│   └── processed/
├── config/
├── etl/
├── models/
├── notebooks/
├── simulation/
├── dashboard/        ← in progress
└── logs/
```

## Related Repository
| Part | Description | Link |
|------|-------------|------|
| Part 1 — Pipeline | Data engineering, ETL, forecasting models | You are here |
| Part 2 — Agent | Agentic AI and demand intelligence module | [perishable-retail-demand-agent](YOUR_LINK_HERE) |
