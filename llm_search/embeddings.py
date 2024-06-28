from typing import Tuple
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def load_model_and_tokenizer() -> Tuple[SentenceTransformer, AutoTokenizer, int, str]:
    """Load the model and tokenizer, caching"""
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device=device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = tokenizer.model_max_length
    return model, tokenizer, max_length, device

def get_embeddings(texts: list[str]) -> torch.Tensor:
    """Get embeddings for a list of texts"""
    model, _, _, device = load_model_and_tokenizer()
    embeddings = model.encode(texts, device=device, show_progress_bar=False)
    return embeddings

def split_into_chunks(text: str, max_length_tokens: int) -> list[str]:
    """Split text into chunks of max_length_tokens for processing by the model"""
    _, tokenizer, _, _ = load_model_and_tokenizer()
    encoded_input = tokenizer.encode(text, truncation=False)
    max_tokens = max_length_tokens - tokenizer.num_special_tokens_to_add()
    chunks = [encoded_input[i:i + max_tokens] for i in range(0, len(encoded_input), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def get_document_embeddings(text: str) -> torch.Tensor:
    """Get embeddings for a document"""
    _, _, max_length, _ = load_model_and_tokenizer()
    chunks = split_into_chunks(text, max_length)
    embeddings = get_embeddings(chunks)
    return embeddings