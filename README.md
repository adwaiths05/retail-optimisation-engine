# Retail Optimization and Personalization Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX-005BEA?style=flat&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)

Relevance-first recommendation system built on Instacart basket data, with live event feedback, experiment tracking, drift monitoring, and dynamic model loading.

## 2-Minute Story

1. Base behavior data comes from Instacart orders.
2. A two-tower retriever creates a candidate set quickly.
3. A feature-rich ML reranker scores candidates using user-product frequency, recency, reorder ratio, embedding similarity, and category affinity.
4. Business signals (margin and inventory) are secondary boosts, not primary rank drivers.
5. Streamlit dashboard logs view, click, cart_add, and purchase events.
6. Those events feed A/B metrics and Evidently drift windows.
7. MLflow tracks experiments and artifacts; promotion compares candidate vs current model before publishing metadata.

## What Is Real vs Simulated

- Real:
	- Event logging pipeline (view/click/cart_add/purchase)
	- Retrieval and reranking inference
	- Redis caching for embeddings and assignment state
	- API latency monitoring and model metadata loading
	- Drift monitoring against recent event windows
- Simulated:
	- Margin/inventory business features and their scale
	- Traffic scale and retraining trigger orchestration
	- Some pricing/economic signals used for portfolio demonstration

## Architecture

```mermaid
flowchart LR
		U[Dashboard User] --> UI[Streamlit Dashboard]
		UI --> API[FastAPI API]

		API --> EMB[User Tower ONNX]
		EMB --> RET[pgvector Retrieval]
		RET --> RR[ML Reranker\nXGBoost rank:ndcg]
		RR --> RES[Top-K Recommendations]

		UI --> EVT[Event API\nview/click/cart_add/purchase]
		EVT --> DB[(Postgres)]

		RR --> REDIS[(Redis Cache)]
		API --> META[Model Metadata]

		DB --> AB[A/B Metrics]
		DB --> DRIFT[Evidently Drift\ntraining vs recent events]
		DRIFT --> UI

		TRAIN[Training + Offline Eval] --> MLF[MLflow Tracking]
		MLF --> GATE[Promotion Gate\ncompare candidate vs current]
		GATE --> META
```

## Ranking Approach

- Retrieval stage: Two-tower embeddings retrieve top-N nearest products.
- Reranking stage: XGBoost ranker optimized with ranking objective (`rank:ndcg`) and grouped user lists.
- Relevance features:
	- user_product_frequency
	- user_product_recency
	- user_product_reorder_ratio
	- embedding_similarity
	- category_affinity
- Business controls:
	- margin and inventory are low-weight boosts.
	- default group weights prioritize relevance:
		- control: relevance 0.92, margin 0.05, inventory 0.03
		- margin_boost: relevance 0.86, margin 0.10, inventory 0.04

## Offline Credibility Metrics

Run staged offline evaluation:

```bash
python training/offline_eval.py
```

Outputs three comparable blocks:

- retrieval_only
- retrieval_plus_heuristic_rerank
- retrieval_plus_ml_rerank

Metrics written to:

- mlops/reports/offline_ranking_metrics.json

## Demo Flow

1. Open dashboard.
2. Generate recommendations for a user.
3. Trigger product events in order: view -> click -> cart_add -> purchase.
4. Refresh recommendations.
5. Open experiment analytics for CTR, click-to-cart, and conversion.
6. Run observability tab drift report.
7. Verify live model version in model ops tab.

## MLflow and Promotion

Use MLflow for experiment tracking and artifact versioning:

```bash
python mlops/train_with_mlflow.py
```

Promotion gate (latency, parity, and not-worse-than-current) updates metadata consumed by the API:

```bash
python mlops/publish_model.py
```

## Tooling: One-Line Purpose

- FastAPI: low-latency async inference and event APIs.
- PostgreSQL + pgvector: vector retrieval and event persistence.
- Redis: cache embeddings and stable A/B assignment.
- XGBoost: fast interpretable reranking model.
- ONNX Runtime: efficient serving for embedding/rerank models.
- MLflow: experiment runs, metrics, and artifact lineage.
- Evidently: drift reporting between training and live event windows.
- Streamlit: demo/control UI for interactions and monitoring.

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Configure environment.

```bash
cp .env.example .env
```

3. Start services.

```bash
docker compose -f deployments/docker-compose.yml up --build
```

4. Open:

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Project Layout

```text
src/api         FastAPI routes and middleware
src/engine      Retrieval, reranking, session policy
training        Retrieval/reranker training and offline eval
mlops           Tracking, promotion, drift workflows
models          ONNX, reranker bundles, metadata
src/frontend    Streamlit dashboard and demo controls
```