import os
import json
import faiss
import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

# Import the enhanced setups
try:
    from .contextual_vector_setup import ContextualVectorSetup
    HAS_CONTEXTUAL = True
except ImportError:
    HAS_CONTEXTUAL = False

try:
    from .anthropic_vector_setup import AnthropicVectorSetup, main_anthropic_setup
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

def parse_pdfs_to_docs(data_dir: str = "data/annual_reports") -> list:
    converter = DocumentConverter()
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            print(f"Processing {filename}...")
            result = converter.convert(path)
            
            # Split document into chunks by paragraphs or sections
            doc_text = result.document.export_to_markdown()
            
            # Split into chunks (simple approach - can be improved)
            chunks = doc_text.split('\n\n')
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    docs.append({
                        "content": chunk.strip(),
                        "metadata": {
                            "source_name": filename,
                            "source_type": "pdf",
                            "chunk_number": i + 1
                        }
                    })
    return docs

def generate_embeddings(docs: list) -> list:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([doc["content"] for doc in docs], convert_to_tensor=True)
    for doc, embedding in zip(docs, embeddings):
        doc["embedding"] = embedding.tolist()
    return docs

def store_and_deploy_vector_db(docs: list, index_path: str = "data/vector_db/vector_index.faiss") -> bool:
    embeddings = np.array([doc["embedding"] for doc in docs])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    
    # Store both content and metadata for each document
    metadata = []
    for doc in docs:
        metadata.append({
            "content": doc["content"],
            **doc["metadata"]
        })
    
    with open("data/vector_db/vector_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Verify deployment
    if not os.path.exists(index_path) or not os.path.exists("data/vector_db/vector_metadata.json"):
        raise RuntimeError("Vector DB deployment failed: files not created.")
    print(f"Vector DB deployed successfully at {index_path}")
    return True

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    use_anthropic = "--anthropic" in sys.argv and HAS_ANTHROPIC
    use_contextual = "--contextual" in sys.argv and HAS_CONTEXTUAL
    
    if use_anthropic:
        print("🧠 Using Anthropic Claude models for embeddings...")
        success = main_anthropic_setup(use_contextual=True)
    elif use_contextual:
        print("🚀 Using enhanced contextual vector setup...")
        from .contextual_vector_setup import main as contextual_main
        success = contextual_main(use_contextual=True)
    else:
        print("📄 Using standard vector setup...")
        docs = parse_pdfs_to_docs()
        docs_with_embeddings = generate_embeddings(docs)
        success = store_and_deploy_vector_db(docs_with_embeddings)
    
    if success:
        print("✅ PDF parsing and vector DB deployment completed.")
    
    print("\n💡 Available options:")
    print("   --anthropic    : Use Claude models for embeddings (recommended)")
    print("   --contextual   : Use sentence-transformers with contextual enhancement")
    print("   (no flags)     : Use standard sentence-transformers")