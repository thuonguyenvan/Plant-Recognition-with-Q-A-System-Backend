# Plant Medicine RAG Backend - Project Complete!

## 🎉 PROJECT OVERVIEW

Vietnamese Plant Medicine Q&A system với 3 flows:
- **Flow 1:** Image-only classification
- **Flow 2:** Image + Text Q&A với LLM routing
- **Flow 3:** Pure text RAG

## 🏗️ ARCHITECTURE

```
Frontend (Streamlit)
        ↓
FastAPI Backend
        ↓
    ┌───┴───────────────────┐
    │                       │
CV API          OG-RAG HyperGraph
(Image)         (Supabase + pgvector)
                       ↓
               Vietnamese Embeddings
               (AITeamVN, 1024-dim)
                       ↓
                  MegLLM API
              (OpenAI-compatible)
```

## ✅ IMPLEMENTED FEATURES

### Core Services
- ✅ Vietnamese embedding service (AITeamVN/Vietnamese_Embedding)
- ✅ Supabase vector database (9,954 hypernodes)
- ✅ CV API client (plant classification)
- ✅ MegLLM client (OpenAI SDK)
- ✅ OG-RAG query engine

### Data Processing
- ✅ JSON-LD loader (1,305 plants)
- ✅ Key normalizer (80+ mappings)
- ✅ Value chunker (250 tokens, sentence-level)
- ✅ Ontology flattener (7,417 facts)

### Flows
- ✅ Flow 1: Top-5 predictions + summaries
- ✅ Flow 2: LLM routing + full plant context
- ✅ Flow 3: Pure RAG with sources

### API
- ✅ FastAPI with CORS
- ✅ 8 endpoints (classify, detail, ask, health)
- ✅ File upload + URL support

## 📊 DATABASE STATS

| Metric | Value |
|--------|-------|
| Plants | 1,305 |
| Facts | 7,417 |
| HyperNodes | 9,954 |
| Embedding dim | 1024 |
| Vector search | ✅ Working |

## 🚀 QUICK START

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup .env
```bash
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
MEGLLM_API_KEY=your_key
```

### 3. Run API
```bash
python main.py
# or
uvicorn main:app --reload
```

### 4. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Flow 3 (RAG)
curl -X POST http://localhost:8000/api/flow3/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Cây nào chữa ho?", "top_k": 5}'
```

## 📁 PROJECT STRUCTURE

```
RAG_BACKEND/
├── main.py              # FastAPI app
├── config.py            # Settings
├── requirements.txt     # Dependencies
├── .env                # Credentials
│
├── api/                # (placeholder)
├── services/           # Core services
│   ├── embedding_service.py
│   ├── vector_db_service.py
│   ├── cv_api_client.py
│   ├── llm_client.py
│   ├── ograg_engine.py
│   ├── flow1_service.py
│   ├── flow2_service.py
│   └── flow3_service.py
│
├── utils/              # Utilities
│   ├── data_loader.py
│   ├── key_normalizer.py
│   └── chunker.py
│
├── scripts/            # Data processing
│   ├── flatten_ontology.py
│   ├── build_hypergraph.py
│   ├── import_embeddings.py
│   └── clean_duplicates.py
│
├── tests/              # Tests
│   ├── test_connection.py
│   └── test_hypergraph.py
│
└── data/               # JSON-LD files
    └── ontology_node_*.jsonld
```

## 🔧 CONFIGURATION

### config.py
- Supabase credentials
- MegLLM API key
- CV API endpoint
- Model settings

### Vector Search Optimization
- Default top_k: 10 (reduced for performance)
- Threshold: 0.4 (lowered from 0.5)
- Retry logic: 2 attempts with adaptive top_k
- Timeout: 120s

## 📝 API ENDPOINTS

### Flow 1: Image Classification
- `POST /api/flow1/classify` - Upload image
- `POST /api/flow1/classify-url` - Image URL
- `GET /api/flow1/detail/{class_name}` - Plant details

### Flow 2: Image + Text Q&A
- `POST /api/flow2/ask` - Upload + question
- `POST /api/flow2/ask-url` - URL + question

### Flow 3: Pure RAG
- `POST /api/flow3/ask` - Text question

### System
- `GET /` - Basic health
- `GET /health` - Detailed health

## ⚠️ KNOWN LIMITATIONS

1. **Supabase Free Tier:**
   - Memory limit: 32MB (can't rebuild indexes)
   - Statement timeout (handled with retry)

2. **Vector Search:**
   - Works but requires retry logic
   - Optimized with reduced top_k

3. **MegLLM:**
   - Using OpenAI-compatible endpoint
   - Model: openai-gpt-oss-120b

## 🎯 NEXT STEPS (Optional)

1. **Streamlit Demo** - Visual UI for all 3 flows
2. **ChromaDB Migration** - Alternative to Supabase (no timeout)
3. **Caching** - Redis for frequent queries
4. **Batch Processing** - Background jobs for embeddings
5. **Monitoring** - Logging + metrics

## 📚 DEPENDENCIES

```
fastapi
uvicorn
python-multipart
sentence-transformers
torch
supabase
python-dotenv
pydantic-settings
httpx
openai
tqdm
numpy
```

## 🤝 INTEGRATION EXAMPLES

### Python
```python
import requests

# Flow 3 RAG
response = requests.post(
    "http://localhost:8000/api/flow3/ask",
    json={"question": "Sâm cau có tác dụng gì?"}
)
print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:8000/api/flow3/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Cây nào trị ho?"}'
```

## ✨ HIGHLIGHTS

- ✅ **Vector search working** với retry mechanism
- ✅ **9,954 nodes indexed** trong Supabase
- ✅ **OG-RAG hypergraph** fully functional
- ✅ **3 complete flows** implemented
- ✅ **Production-ready API** with error handling
- ✅ **OpenAI-compatible LLM** integration

---

**Status:** ✅ COMPLETE & READY FOR DEMO
**Time:** ~4 hours from start to finish  
**Next:** Build Streamlit UI or test with real queries!
