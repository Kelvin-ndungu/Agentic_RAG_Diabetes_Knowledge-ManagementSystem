"""
Jina Embedding Function for ChromaDB.
"""
import os
import requests
import time
from typing import List


class JinaEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using Jina API.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "jina-embeddings-v4",
        task: str = "text-matching",
        api_url: str = "https://api.jina.ai/v1/embeddings",
        batch_size: int = 10,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY environment variable is required. "
                "Set it with: export JINA_API_KEY=your_api_key"
            )
        self.model = model
        self.task = task
        self.api_url = api_url
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
    
    def name(self) -> str:
        return "jina-embeddings-v4"
    
    def __call__(self, input):
        """Generate embeddings for input text(s)."""
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
        
        if not texts:
            return []
        
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using Jina API."""
        data = {
            "model": self.model,
            "task": self.task,
            "input": [{"text": text} for text in texts]
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings = []
                if 'data' in result:
                    for item in result['data']:
                        if 'embedding' in item:
                            embeddings.append(item['embedding'])
                    return embeddings
                else:
                    raise ValueError(f"Unexpected API response format: {result}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠ API request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to get embeddings after {self.max_retries} attempts: {e}")
        
        return []

