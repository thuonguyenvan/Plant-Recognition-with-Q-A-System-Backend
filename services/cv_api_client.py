"""
CV API Client for Plant Classification
Integrates with thuonguyenvan-plantsclassify.hf.space
"""
import httpx
from typing import List, Dict, Optional
import time


class CVAPIClient:
    """Client for plant classification API"""
    
    def __init__(
        self,
        base_url: str = "https://thuonguyenvan-plantsclassify.hf.space",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize CV API client
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout)
    
    def classify_image(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """
        Classify plant image
        
        Args:
            image_path: Path to local image file (for upload)
            image_url: URL to image (for URL prediction)
            
        Returns:
            List of predictions with class_name and confidence
            
        Example:
            [
                {"class_name": "Curculigo_orchioides", "confidence": 0.85},
                ...
            ]
        """
        if not image_path and not image_url:
            raise ValueError("Either image_path or image_url must be provided")
        
        if image_path and image_url:
            raise ValueError("Provide either image_path or image_url, not both")
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                if image_path:
                    return self._classify_from_file(image_path)
                else:
                    return self._classify_from_url(image_url)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def _classify_from_file(self, image_path: str) -> List[Dict[str, float]]:
        """Classify from uploaded file"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.client.post(
                f"{self.base_url}/predict/upload",
                files=files
            )
            response.raise_for_status()
            return response.json()['predictions']
    
    def _classify_from_url(self, image_url: str) -> List[Dict[str, float]]:
        """Classify from image URL"""
        response = self.client.post(
            f"{self.base_url}/predict/url",
            json={"url": image_url}
        )
        response.raise_for_status()
        return response.json()['predictions']
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def __del__(self):
        """Cleanup client"""
        self.client.close()


# Singleton instance
_cv_api_client = None

def get_cv_api_client() -> CVAPIClient:
    """Get cached CV API client"""
    global _cv_api_client
    if _cv_api_client is None:
        from config import get_settings
        settings = get_settings()
        _cv_api_client = CVAPIClient(
            base_url=settings.cv_api_url,
            timeout=settings.cv_api_timeout
        )
    return _cv_api_client
