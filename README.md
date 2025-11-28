# 🌿 Plant Recognition with Q&A System - Backend

FastAPI backend for Vietnamese plant recognition and Q&A using RAG (Retrieval-Augmented Generation) with OG-RAG hypergraph architecture.

## 🎯 Features

- **Flow 1:** Image-only plant classification (Top-5 predictions)
- **Flow 2:** Image + Question (Plant identification → Contextual Q&A)
- **Flow 3:** Text-only Q&A (Pure RAG with Vietnamese embeddings)

## 🏗️ Tech Stack

- **API:** FastAPI + Uvicorn
- **Database:** Supabase (PostgreSQL + pgvector)
- **Embeddings:** Vietnamese-Embedding (1024-dim)
- **LLM:** MegaLLM API (qwen/qwen3-next-80b-a3b-instruct)
- **CV Model:** Plant Classification API
- **Architecture:** OG-RAG Hypergraph (9,954 nodes, 1,305 plants)

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Supabase account
- MegaLLM API key
- Plant Classification API endpoint

### 2. Installation

```bash
# Clone repository
git clone https://github.com/thuonguyenvan/Plant-Recognition-with-Q-A-System-Backend.git
cd Plant-Recognition-with-Q-A-System-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required environment variables:**

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key

# MegaLLM
MEGLLM_API_KEY=your_megallm_api_key

# Computer Vision API (optional - has default)
CV_API_URL=https://your-cv-api-endpoint

# Optional: Direct DB connection for data import scripts
SUPABASE_DB_URI=postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
```

> **Note:** `EMBEDDING_MODEL_NAME` has a default value (`AITeamVN/Vietnamese_Embedding`) and doesn't need to be set unless you want to use a different model.

### 4. Database Setup

Run the SQL setup script in your Supabase SQL Editor:

```bash
# Copy content from set_up_supabasedb.sql
# Paste and run in: https://app.supabase.com/project/_/sql
```

### 5. Import Data (Optional)

If you have the data files:

```bash
# Import hypernodes with embeddings
python scripts/fast_import.py --embeddings plant_hypernodes_with_embeddings.json
```

> **Note:** Large data files are not included in this repository. Contact maintainer for access.

### 6. Run Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python main.py
```

Server will start at: **http://localhost:8000**

---

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

### Flow 1: Image Classification

```bash
# Upload image file
POST /api/flow1/classify
Content-Type: multipart/form-data
Body: file=<image>

# Or use image URL
POST /api/flow1/classify-url
Content-Type: application/json
Body: {"image_url": "https://..."}

# Get plant details
GET /api/flow1/detail/{plant_name}
```

### Flow 2: Image + Question

```bash
# Upload image + question
POST /api/flow2/identify
Content-Type: multipart/form-data
Body: file=<image>

# Then ask question about identified plant
POST /api/flow2/ask
Content-Type: application/json
Body: {
  "question": "Cây này có tác dụng gì?",
  "plant_name": "Sâm cau"
}
```

### Flow 3: Text Q&A (RAG)

```bash
POST /api/flow3/ask
Content-Type: application/json
Body: {
  "question": "Cây nào chữa ho?",
  "top_k": 10
}
```

---

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test Flow 3 (RAG)
curl -X POST http://localhost:8000/api/flow3/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Sâm cau có tác dụng gì?"}'

# Test Flow 1 (Classification)
curl -X POST http://localhost:8000/api/flow1/classify \
  -F "file=@path/to/plant_image.jpg"
```

---

## 📁 Project Structure

```
Plant-Recognition-with-Q-A-System-Backend/
├── main.py                    # FastAPI application entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── set_up_supabasedb.sql     # Database setup script
│
├── services/                 # Core business logic
│   ├── cv_api_client.py     # Plant classification API client
│   ├── embedding_service.py  # Vietnamese embedding service
│   ├── llm_client.py        # Groq LLM client
│   ├── vector_db_service.py # Supabase vector operations
│   ├── ograg_engine.py      # OG-RAG hypergraph engine
│   ├── query_reformulator.py # Query enhancement
│   ├── flow1_service.py     # Image classification flow
│   ├── flow2_service.py     # Image + Q&A flow
│   └── flow3_service.py     # Text Q&A flow
│
├── utils/                    # Utility modules
│   ├── data_loader.py       # JSON-LD ontology loader
│   ├── key_normalizer.py    # Attribute name mapping
│   └── chunker.py           # Text chunking utilities
│
├── scripts/                  # Data processing scripts
│   ├── flatten_ontology.py  # Convert JSON-LD to facts
│   ├── build_hypergraph.py  # Build hypergraph structure
│   ├── import_embeddings.py # Generate embeddings
│   ├── fast_import.py       # Import to Supabase
│   └── clean_duplicates.py  # Remove duplicate nodes
│
└── tests/                    # Test files
    ├── test_connection.py   # Database connection tests
    └── test_hypergraph.py   # Hypergraph tests
```

---

## 🔧 Configuration

### Vector Search Settings

Default settings in `config.py`:

```python
VECTOR_SEARCH_TOP_K = 10
VECTOR_SEARCH_THRESHOLD = 0.4
VECTOR_SEARCH_TIMEOUT = 120
```

### LLM Settings

```python
LLM_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
LLM_BASE_URL = "https://ai.megallm.io/v1"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000
```

---

## 📊 Database Schema

### Hypernodes Table

```sql
CREATE TABLE hypernodes (
    id BIGSERIAL PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    key_embedding vector(1024),
    value_embedding vector(1024),
    plant_name TEXT NOT NULL,
    section TEXT,
    chunk_id INTEGER DEFAULT 0,
    is_chunked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check Supabase project is not paused
# Verify SUPABASE_URL and SUPABASE_KEY in .env
# Test connection:
python tests/test_connection.py
```

### Vector Search Timeout

```bash
# Reduce top_k in request
# Increase threshold (0.5 instead of 0.4)
# Check Supabase free tier limits
```

### Import Errors

```bash
# Ensure python-dotenv is installed
pip install python-dotenv

# Check .env file exists and has correct format
```

---

## 📚 Documentation

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **ReDoc:** http://localhost:8000/redoc
- **CV API Docs:** See `CV_API_DOCS.md`
- **Flow 2 API:** See `FLOW2_API.md`
- **Kaggle Embedding Guide:** See `KAGGLE_EMBEDDING_GUIDE.md`

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Thuong Nguyen Van** - [@thuonguyenvan](https://github.com/thuonguyenvan)

---

## 🙏 Acknowledgments

- **OG-RAG Paper:** [Ontology-Grounded RAG](https://arxiv.org/html/2412.15235v1)
- **Vietnamese Embedding:** AITeamVN/Vietnamese_Embedding
- **Supabase:** Vector database with pgvector
- **MegaLLM:** OpenAI-compatible LLM API

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Email: thuongnguyenvan2209@gmail.com

---

**Status:** ✅ Production Ready  
**Last Updated:** November 2025
