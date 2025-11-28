"""
Main FastAPI Application
Plant Medicine RAG Backend with 3 Flows
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import os
import glob

from config import get_settings
from services.cv_api_client import get_cv_api_client
from services.llm_client import get_megllm_client
from services.embedding_service import get_embedding_service
from services.vector_db_service import SupabaseVectorDB
from services.ograg_engine import get_og_rag_engine
from services.flow1_service import get_flow1_service
from services.flow2_service import get_flow2_service
from services.flow3_service import get_flow3_service
from services.query_reformulator import get_query_reformulator
from utils.data_loader import get_plant_data_loader

# Initialize app
app = FastAPI(
    title="Plant Medicine RAG API",
    description="Vietnamese Plant Medicine Q&A with 3 flows: Image Only, Image+Text, Pure RAG",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy)
settings = get_settings()
cv_client = None
llm_client = None
embed_service = None
vector_db = None
og_rag = None
data_loader = None
reformulator = None
flow1 = None
flow2 = None
flow3 = None


def init_services():
    """Initialize all services"""
    global cv_client, llm_client, embed_service, vector_db, og_rag, data_loader, reformulator, flow1, flow2, flow3
    
    if cv_client is None:
        print("Initializing services...")
        cv_client = get_cv_api_client()
        llm_client = get_megllm_client()
        embed_service = get_embedding_service()
        vector_db = SupabaseVectorDB(
            url=settings.supabase_url,
            key=settings.supabase_anon_key,
            timeout=120
        )
        og_rag = get_og_rag_engine(embed_service, vector_db)
        data_loader = get_plant_data_loader()
        reformulator = get_query_reformulator(llm_client)
        
        # Initialize flow services
        flow1 = get_flow1_service(cv_client, data_loader)
        flow2 = get_flow2_service(cv_client, llm_client, og_rag, data_loader)
        flow3 = get_flow3_service(llm_client, og_rag, reformulator)
        
        print("✅ All services initialized")


# Request/Response models
class ImageURLRequest(BaseModel):
    image_url: str


class Flow2Request(BaseModel):
    question: str
    image_url: Optional[str] = None


class Flow3Request(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = []
    top_k: Optional[int] = 8
    selected_plant: Optional[str] = None  # NEW: Track selected plant from frontends


# Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    init_services()


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Plant Medicine RAG API",
        "flows": ["flow1", "flow2", "flow3"]
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    init_services()
    return {
        "status": "healthy",
        "services": {
            "cv_api": cv_client.health_check() if cv_client else False,
            "vector_db": vector_db.count_nodes() if vector_db else 0,
            "data_loader": data_loader.count_plants() if data_loader else 0
        }
    }


# FLOW 1: Image Only
@app.post("/api/flow1/classify")
async def flow1_classify_upload(file: UploadFile = File(...)):
    """
    Flow 1: Classify plant from uploaded image
    Returns top-5 predictions with summaries
    """
    init_services()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Classify
        result = flow1.classify_and_summarize(image_path=tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flow1/classify-url")
async def flow1_classify_url(request: ImageURLRequest):
    """
    Flow 1: Classify plant from image URL
    """
    init_services()
    
    try:
        result = flow1.classify_and_summarize(image_url=request.image_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/flow1/detail/{class_name}")
async def flow1_get_detail(class_name: str):
    """
    Flow 1: Get detailed plant information
    """
    init_services()
    
    try:
        result = flow1.get_plant_detail(class_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FLOW 2: Image + Text Q&A (Two-Step Process)

@app.post("/api/flow2/identify")
async def flow2_identify(file: UploadFile = File(...)):
    """
    Flow 2 - Step 1: Identify plant from image only
    Returns predictions for user to select
    """
    init_services()
    
    try:
        print(f"[Flow2/Identify] File: {file.filename}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[Flow2/Identify] Image saved to: {tmp_path}")
        
        # Identify plant only
        result = flow2.identify_plant(image_path=tmp_path, top_k=5)
        
        print(f"[Flow2/Identify] Found {len(result.get('predictions', []))} predictions")
        
        # Cleanup
        os.unlink(tmp_path)
        
        return result
    except Exception as e:
        import traceback
        print(f"[Flow2/Identify ERROR] {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flow2/ask")
async def flow2_ask_with_plant(
    question: str,
    selected_plant: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """
    Flow 2 - Step 2: Answer question about selected plant
    
    Supports two modes:
    1. Two-step: selected_plant provided (after /identify)
    2. One-step: file provided (legacy - auto-identify + answer)
    """
    init_services()
    
    try:
        print(f"[Flow2/Ask] Question: {question}")
        print(f"[Flow2/Ask] Selected plant: {selected_plant}")
        print(f"[Flow2/Ask] File: {file.filename if file else None}")
        
        # Mode 1: User selected plant (two-step)
        if selected_plant:
            result = flow2.answer_with_plant(
                question=question,
                plant_class_name=selected_plant,
                use_rag=True
            )
            print(f"[Flow2/Ask] Answered with selected plant")
            return result
        
        # Mode 2: Legacy one-step (auto-identify from image)
        elif file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            print(f"[Flow2/Ask] Image saved to: {tmp_path}")
            
            # One-step answer
            result = flow2.answer_question(
                question=question,
                image_path=tmp_path
            )
            
            print(f"[Flow2/Ask] Completed one-step flow")
            
            # Cleanup
            os.unlink(tmp_path)
            
            return result
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'selected_plant' or 'file' must be provided"
            )
        
    except Exception as e:
        import traceback
        print(f"[Flow2/Ask ERROR] {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoint - kept for backward compatibility
@app.post("/api/flow2/ask-legacy")
async def flow2_ask_upload(question: str, file: UploadFile = File(...)):
    """
    Flow 2: Ask question about plant in uploaded image (legacy one-step)
    """
    init_services()
    
    try:
        print(f"[Flow2] Received question: {question}")
        print(f"[Flow2] File: {file.filename}, content_type: {file.content_type}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[Flow2] Image saved to: {tmp_path}")
        
        # Answer question
        result = flow2.answer_question(
            question=question,
            image_path=tmp_path
        )
        
        print(f"[Flow2] Successfully processed question")
        
        # Cleanup
        os.unlink(tmp_path)
        
        return result
    except Exception as e:
        import traceback
        print(f"[Flow2 ERROR] Exception occurred:")
        print(f"[Flow2 ERROR] Type: {type(e).__name__}")
        print(f"[Flow2 ERROR] Message: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flow2/ask-url")
async def flow2_ask_url(request: Flow2Request):
    """
    Flow 2: Ask question about plant from image URL
    """
    init_services()
    
    try:
        result = flow2.answer_question(
            question=request.question,
            image_url=request.image_url
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flow3/ask")
async def flow3_ask(request: Flow3Request):
    """
    Flow 3: Pure RAG Q&A with Query Reformulation
    
    Handles text-only questions using:
    - Intelligent query reformulation based on conversation context
    - Plant context from previous selections
    - Intent detection (chitchat, comparison, specific, generic)
    """
    init_services()
    
    try:
        result = flow3.answer_question(
            question=request.question,
            top_k=request.top_k,
            conversation_history=request.conversation_history,
            selected_plant=request.selected_plant  # NEW: Pass selected plant
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#GET plant images
@app.get("/api/plants/{class_name}/images")
async def get_plant_images(class_name: str, request: Request):
    """
    Get all images for a specific plant class
    """
    image_dir = f"inat_representative_photos/{class_name}"
    
    if not os.path.exists(image_dir):
        return {"class_name": class_name, "image_urls": []}
    
    # Get all jpg images
    images = glob.glob(f"{image_dir}/*.jpg")
    images.sort()
    
    # Build URLs with dynamic base URL
    base_url = str(request.base_url).rstrip('/')
    image_urls = [f"{base_url}/plant-images/{class_name}/{os.path.basename(img)}" for img in images]
    
    return {
        "class_name": class_name,
        "image_urls": image_urls,
        "count": len(image_urls)
    }


# Mount static files AFTER all routes
app.mount("/plant-images", StaticFiles(directory="inat_representative_photos"), name="plant-images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
