import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def parse_pdfs_to_docs_simple(data_dir: str = "data/annual_reports") -> list:
    """
    Simple fallback PDF parser using basic text extraction.
    For demo purposes - creates mock data if PDF parsing fails.
    """
    docs = []
    
    # For now, let's create some sample sustainability data
    # This simulates what would be extracted from the PDFs
    sample_docs = [
        {
            "content": "Our company achieved a 25% reduction in carbon emissions compared to the previous year. This was accomplished through energy efficiency improvements and renewable energy adoption.",
            "metadata": {
                "source_name": "sustainability_report_2024.pdf",
                "source_type": "pdf",
                "chunk_number": 1,
                "topic": "carbon_emissions"
            }
        },
        {
            "content": "Water consumption decreased by 15% through implementation of water recycling systems and improved process efficiency.",
            "metadata": {
                "source_name": "sustainability_report_2024.pdf", 
                "source_type": "pdf",
                "chunk_number": 2,
                "topic": "water_management"
            }
        },
        {
            "content": "Employee diversity and inclusion programs resulted in a 30% increase in female leadership positions across all business units.",
            "metadata": {
                "source_name": "sustainability_report_2024.pdf",
                "source_type": "pdf", 
                "chunk_number": 3,
                "topic": "diversity_inclusion"
            }
        },
        {
            "content": "Supply chain sustainability initiatives included working with 85% of key suppliers to establish environmental standards and monitoring programs.",
            "metadata": {
                "source_name": "sustainability_report_2024.pdf",
                "source_type": "pdf",
                "chunk_number": 4,
                "topic": "supply_chain"
            }
        },
        {
            "content": "Renewable energy now accounts for 60% of our total energy consumption, with solar and wind installations at major facilities.",
            "metadata": {
                "source_name": "sustainability_report_2024.pdf",
                "source_type": "pdf",
                "chunk_number": 5,
                "topic": "renewable_energy"
            }
        }
    ]
    
    print(f"Created {len(sample_docs)} sample document chunks for testing")
    return sample_docs

def generate_embeddings(docs: list) -> list:
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    contents = [doc["content"] for doc in docs]
    embeddings = model.encode(contents, convert_to_tensor=False)
    
    for doc, embedding in zip(docs, embeddings):
        doc["embedding"] = embedding.tolist()
    
    return docs

def store_and_deploy_vector_db(docs: list, index_path: str = "data/vector_db/vector_index.faiss") -> bool:
    print("Creating FAISS index...")
    embeddings = np.array([doc["embedding"] for doc in docs]).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print("Saving vector database files...")
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
    
    print(f"✅ Vector DB deployed successfully!")
    print(f"   Index: {index_path}")
    print(f"   Metadata: data/vector_db/vector_metadata.json")
    print(f"   Total documents: {len(docs)}")
    print(f"   Embedding dimension: {dimension}")
    return True

if __name__ == "__main__":
    print("Setting up vector database with sample data...")
    docs = parse_pdfs_to_docs_simple()
    docs_with_embeddings = generate_embeddings(docs)
    result = store_and_deploy_vector_db(docs_with_embeddings)