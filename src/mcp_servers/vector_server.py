#!/usr/bin/env python3
"""
Vector database MCP server for similarity search over sustainability documents.
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("vector-server")

# Initialize vector database components globally
vector_db_path = "data/vector_db/vector_index.faiss"
metadata_path = "data/vector_db/vector_metadata.json"

if os.path.exists(vector_db_path) and os.path.exists(metadata_path):
    index = faiss.read_index(vector_db_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.info(f"Loaded vector DB with {len(metadata)} documents")
    
    # Detect embedding type from metadata
    embedding_model_used = None
    if metadata and "embedding_model" in metadata[0]:
        embedding_model_used = metadata[0]["embedding_model"]
        logger.info(f"Detected embedding model: {embedding_model_used}")
else:
    logger.error(f"Vector database not found at {vector_db_path}")
    index = None
    metadata = []
    embedding_model_used = None

# Initialize embedding models
model = SentenceTransformer('all-MiniLM-L6-v2')  # For sentence-transformers embeddings

# For Anthropic embeddings
try:
    from anthropic import Anthropic
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    HAS_ANTHROPIC = True
except ImportError:
    anthropic_client = None
    HAS_ANTHROPIC = False

def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for query using the same method as stored embeddings."""
    global embedding_model_used, model, anthropic_client
    
    if embedding_model_used and "claude" in embedding_model_used.lower() and HAS_ANTHROPIC:
        # Use Anthropic embedding
        logger.info(f"Generating Anthropic embedding for query using {embedding_model_used}")
        
        prompt = f"""You are a semantic embedding generator. Convert the following text into a numerical vector representation that captures its semantic meaning.

Text to embed:
{query}

Generate exactly 512 floating point numbers between -1.0 and 1.0 that represent the semantic meaning of this text. Focus on:
1. Key concepts and entities
2. Sentiment and tone
3. Topic classification
4. Semantic relationships
5. Context and domain

Return ONLY a JSON array of 512 numbers, nothing else:"""

        try:
            response = anthropic_client.messages.create(
                model=embedding_model_used,
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            embedding_text = response.content[0].text.strip()
            embedding = json.loads(embedding_text)
            
            # Validate and normalize
            if len(embedding) != 512:
                if len(embedding) < 512:
                    embedding.extend([0.0] * (512 - len(embedding)))
                else:
                    embedding = embedding[:512]
            
            # Normalize
            embedding = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"Failed to generate Anthropic embedding: {e}, falling back to sentence-transformers")
    
    # Use sentence-transformers (default)
    logger.info("Generating sentence-transformers embedding for query")
    query_embedding = model.encode([query], convert_to_tensor=False)
    return np.array(query_embedding).astype('float32')

@mcp.tool(
    name="vector_search",
    description="Search for relevant sustainability documents using enhanced semantic similarity with contextual embeddings"
)
def vector_search(query: str, k: int = 3) -> str:
    """
    Search for documents related to the query using vector similarity.
    
    Args:
        query: The search query
        k: Number of top results to return (default: 3)
    
    Returns:
        JSON string with the search results
    """
    if index is None or not metadata:
        return json.dumps({
            "error": "Vector database not initialized",
            "results": []
        })
    
    try:
        # Generate embedding for the query using the same method as stored embeddings
        query_embedding = generate_query_embedding(query)
        
        # Search the index
        distances, indices = index.search(query_embedding, min(k, len(metadata)))
        
        # Prepare results with enhanced metadata
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(metadata):  # Valid index
                doc = metadata[idx].copy()
                doc['similarity_score'] = float(1 / (1 + distance))  # Convert L2 distance to similarity
                doc['rank'] = i + 1
                
                # Add contextual information if available
                if 'context' in doc and doc['context']:
                    doc['has_contextual_info'] = True
                else:
                    doc['has_contextual_info'] = False
                
                results.append(doc)
        
        return json.dumps({
            "query": query,
            "num_results": len(results),
            "results": results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return json.dumps({
            "error": f"Search failed: {str(e)}",
            "results": []
        })

@mcp.tool(
    name="rebuild_vector_db",
    description="Rebuild the vector database with enhanced embeddings (Anthropic Claude, contextual, or standard)"
)
def rebuild_vector_db(method: str = "anthropic", use_contextual: bool = True) -> str:
    """
    Rebuild the vector database with different embedding methods.
    
    Args:
        method: Embedding method ("anthropic", "contextual", or "standard")
        use_contextual: Whether to use contextual embeddings (for anthropic/contextual methods)
    
    Returns:
        JSON string with rebuild status
    """
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        if method == "anthropic":
            from src.util.anthropic_vector_setup import main_anthropic_setup
            success = main_anthropic_setup(use_contextual=use_contextual)
            method_name = f"Anthropic Claude embeddings {'with context' if use_contextual else ''}"
        elif method == "contextual":
            from src.util.contextual_vector_setup import main as contextual_main
            success = contextual_main(use_contextual=True)
            method_name = "contextual embeddings"
        else:
            from src.util.simple_vector_setup import main as simple_main
            success = simple_main()
            method_name = "standard embeddings"
        
        if success:
            # Reload the global variables
            global index, metadata, model
            
            vector_db_path = "data/vector_db/vector_index.faiss"
            metadata_path = "data/vector_db/vector_metadata.json"
            
            if os.path.exists(vector_db_path) and os.path.exists(metadata_path):
                index = faiss.read_index(vector_db_path)
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                return json.dumps({
                    "status": "success",
                    "message": f"Vector database rebuilt successfully using {method_name}",
                    "documents_count": len(metadata),
                    "method": method_name,
                    "embedding_model": metadata[0].get("embedding_model", "unknown") if metadata else "unknown"
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Rebuild completed but files not found"
                })
        else:
            return json.dumps({
                "status": "error", 
                "message": "Failed to rebuild vector database"
            })
            
    except Exception as e:
        logger.error(f"Error rebuilding vector DB: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Rebuild failed: {str(e)}"
        })

if __name__ == "__main__":
    mcp.run()