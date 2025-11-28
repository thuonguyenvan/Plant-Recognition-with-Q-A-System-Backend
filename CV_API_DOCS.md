# 🌿 Plant Classification API - Quick Guide

**API URL:** `https://thuonguyenvan-plantsclassify.hf.space`

---

## Endpoints

### 1️⃣ Upload Image File
```
POST /predict/upload
```

**Example (Python):**
```python
import requests

url = "https://thuonguyenvan-plantsclassify.hf.space/predict/upload"
files = {"file": open("plant.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Example (cURL):**
```bash
curl -X POST https://thuonguyenvan-plantsclassify.hf.space/predict/upload \
  -F "file=@plant.jpg"
```

---

### 2️⃣ Image URL
```
POST /predict/url
Content-Type: application/json
```

**Request:**
```json
{
  "url": "https://example.com/plant.jpg"
}
```

**Example (Python):**
```python
import requests

url = "https://thuonguyenvan-plantsclassify.hf.space/predict/url"
data = {"url": "https://example.com/plant.jpg"}
response = requests.post(url, json=data)
print(response.json())
```

**Example (cURL):**
```bash
curl -X POST https://thuonguyenvan-plantsclassify.hf.space/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/plant.jpg"}'
```

---

## Response Format

```json
{
  "predictions": [
    {
      "class_name": "Oryza_sativa",
      "confidence": 0.8543
    },
    {
      "class_name": "Triticum_aestivum",
      "confidence": 0.0821
    },
    {
      "class_name": "Zea_mays",
      "confidence": 0.0412
    },
    {
      "class_name": "Hordeum_vulgare",
      "confidence": 0.0156
    },
    {
      "class_name": "Setaria_italica",
      "confidence": 0.0068
    }
  ]
}
```

---

## Quick Test

**Health Check:**
```bash
curl https://thuonguyenvan-plantsclassify.hf.space/health
```

**Interactive Docs:**
```
https://thuonguyenvan-plantsclassify.hf.space/docs
```

---

## Notes

- ⏰ Free tier auto-sleeps → first request may take 10-30s
- 📸 Supports: JPEG, PNG
- 🎯 Returns: Top 5 predictions
- 🌱 Classes: 1139 plant species
