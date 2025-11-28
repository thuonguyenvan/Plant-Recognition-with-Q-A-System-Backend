# Kaggle Embedding Generation - Quick Guide

## Bước 1: Chuẩn bị
1. Upload `plant_facts.json` lên Kaggle Dataset
2. Tạo notebook mới trên Kaggle
3. Bật GPU accelerator (Settings → Accelerator → GPU T4 x2)

## Bước 2: Upload Files
- Upload `generate_embeddings_kaggle.ipynb` to Kaggle
- Add `plant_facts.json` as input dataset

## Bước 3: Chạy Notebook
- Chạy tất cả cells
- Thời gian: ~5-10 phút với GPU
- Output: `plant_hypernodes_with_embeddings.json`

## Bước 4: Download & Import
```bash
# Download file từ Kaggle Output
# Đưa vào thư mục RAG_BACKEND

# Import vào Supabase
python scripts/import_embeddings.py --format json --embeddings plant_hypernodes_with_embeddings.json
```

## File Sizes
- Input: `plant_facts.json` (~500KB)
- Output JSON: `plant_hypernodes_with_embeddings.json` (~80MB)
- Output NPZ: `plant_embeddings.npz` + `plant_metadata.json` (~45MB total)

**Recommend:** Dùng JSON format cho đơn giản

## Troubleshooting
- Nếu Kaggle timeout: Giảm batch_size xuống 64
- Nếu Out of Memory: Restart kernel và chạy lại
- Nếu import failed: Check Supabase connection trong .env
