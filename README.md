# Retail Optimization & Personalization Engine 🚀

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX-005BEA?style=flat&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)

A production-grade, real-time retail recommendation and pricing optimization system built using the Instacart dataset. This system is designed for **low-latency (<150ms)**, **margin-aware ranking**, and **automated MLOps deployment**.

---

## 🏗️ System Architecture

The engine uses a multi-stage retrieval and ranking pipeline to balance user relevance with business profitability:

1. **Candidate Retrieval:** A **Two-Tower Neural Network** maps users and 50,000+ products into a 64D embedding space.
2. **Vector Search:** Employs **pgvector** with an **HNSW index** for sub-100ms similarity search.
3. **Real-Time Inference:** User embeddings are generated via an optimized **ONNX** model.
4. **Business-Aware Re-Ranking:** An **XGBoost** classifier scores candidates based on purchase probability, profit margins, and inventory pressure.

---

## 📈 Performance Benchmarks

The system was optimized for high-throughput production environments. Transitioning the User Tower from PyTorch to quantized ONNX yielded significant efficiency gains:

| Metric | PyTorch (Baseline) | ONNX (Optimized) | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Latency** | 0.3340 ms | **0.0527 ms** | **6.34x Faster** |
| **P99 Latency** |  1.8072 ms | **0.1043 ms** | **17.3x More Stable** |
| **Model Size** | 62.61 MB | **12.61 MB** | **80% Smaller** |

**Numerical Parity:** The MAE between PyTorch and ONNX outputs is **6.87e-03**, ensuring zero degradation in recommendation quality.

---

## 🧠 Model Performance (Offline Evaluation)

Evaluated against the Instacart Market Basket dataset using hybrid heuristic baselines:

- **Precision@10:** 0.1672
- **Recall@10:** 0.1949
- **NDCG@10:** 0.2372

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Asynchronous), Uvicorn
- **Database:** PostgreSQL (Neon) with `pgvector`
- **Caching:** Redis (Session-based event storage and embedding cache)
- **ML Frameworks:** PyTorch, XGBoost, ONNX Runtime
- **DevOps:** Docker, Docker Compose, Alembic (Migrations)
- **Frontend:** Streamlit (Intelligence Dashboard)

---

## 🚀 Getting Started

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.11+
- A PostgreSQL (Neon) instance with `pgvector` enabled

### 2. Environment Setup
Clone the repository and create a `.env` file based on the provided template:

```bash
cp .env.example .env
# Update DATABASE_URL and SECRET_KEY in .env
```

### 3. Docker Deployment
Run the entire stack (API + Redis) using Docker Compose:

```bash
docker compose -f deployments/docker-compose.yml up --build
```

The API will be available at `http://localhost:8000` and the Streamlit dashboard at `http://localhost:8501`.

---

## 🔐 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/api/v1/auth/token` | Authenticate and receive JWT token |
| POST | `/api/v1/recommendations` | Get personalized, margin-optimized suggestions |
| POST | `/api/v1/pricing/optimize` | Calculate optimal price based on inventory pressure |
| GET | `/api/v1/experiments/results` | View real-time A/B test performance (Revenue/CTR) |
| GET | `/health` | System vitality and database connection status |

---

## 🧪 A/B Testing & Business Logic

The engine supports real-time experiment simulation. Users are randomly assigned to:

- **Control Group:** Ranked by pure relevance (ML score).
- **Margin Boost Group:** Ranked using a weighted formula:

```
Score = (0.6 * ML_Prob) + (0.3 * Norm_Margin) + (0.1 * Inventory_Pressure)
```

---

## 📂 Project Structure

```plaintext
├── src/
│   ├── api/          # FastAPI routes and middleware
│   ├── core/         # Config and Database connections
│   ├── engine/       # Retrieval, Ranking, and Session logic
│   └── frontend/     # Streamlit UI pages
├── training/         # Model training and ONNX export scripts
├── models/           # Exported ONNX and XGBoost artifacts
├── deployments/      # Dockerfile and start scripts
└── alembic/          # Database migration history
```