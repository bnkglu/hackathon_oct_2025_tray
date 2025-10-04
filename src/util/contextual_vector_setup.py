"""
Enhanced PDF vector setup with contextual retrieval based on Anthropic's guide.
This implementation adds context to chunks before embedding for better retrieval performance.
"""

import os
import json
import faiss
import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from typing import List, Dict, Any
from tqdm import tqdm
import re


class ContextualVectorSetup:
    """Enhanced vector database setup with contextual embeddings."""
    
    def __init__(self):
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def situate_context(self, doc_content: str, chunk: str) -> str:
        """
        Add contextual information to chunks using Claude.
        This helps improve retrieval by providing document context.
        """
        prompt = f"""<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of this chunk. Answer only with the succinct context and nothing else."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Warning: Failed to generate context for chunk: {e}")
            return ""

    def smart_chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Improved chunking strategy with overlap and sentence boundaries.
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap if needed
                if overlap > 0 and len(current_chunk) > overlap:
                    # Take last part of previous chunk as start of new chunk
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def extract_company_info(self, filename: str) -> Dict[str, str]:
        """Extract company information from filename."""
        filename_lower = filename.lower()
        
        if "erste" in filename_lower:
            return {"company": "Erste Group", "sector": "Banking"}
        elif "gsk" in filename_lower:
            return {"company": "GSK", "sector": "Pharmaceuticals"}
        elif "swisscom" in filename_lower:
            return {"company": "Swisscom", "sector": "Telecommunications"}
        else:
            return {"company": "Unknown", "sector": "Unknown"}

    def parse_pdfs_to_docs(self, data_dir: str = "data/annual_reports") -> List[Dict[str, Any]]:
        """
        Parse PDFs into document chunks with improved processing.
        """
        converter = DocumentConverter()
        docs = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf"):
                path = os.path.join(data_dir, filename)
                print(f"Processing {filename}...")
                
                try:
                    result = converter.convert(path)
                    full_content = result.document.export_to_markdown()
                    
                    # Use smart chunking
                    chunks = self.smart_chunk_document(full_content)
                    company_info = self.extract_company_info(filename)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip() and len(chunk.strip()) > 50:  # Filter very short chunks
                            docs.append({
                                "content": chunk.strip(),
                                "full_content": full_content,  # Keep full doc for context
                                "metadata": {
                                    "source_name": filename,
                                    "source_type": "pdf",
                                    "chunk_number": i + 1,
                                    **company_info
                                }
                            })
                    
                    print(f"  → Generated {len([d for d in docs if d['metadata']['source_name'] == filename])} chunks")
                    
                except Exception as e:
                    print(f"  → Error processing {filename}: {e}")
                    continue
        
        return docs

    def generate_contextual_embeddings(self, docs: List[Dict[str, Any]], use_context: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings with optional contextual information.
        """
        contextualized_docs = []
        
        print(f"Generating {'contextual' if use_context else 'standard'} embeddings...")
        
        for doc in tqdm(docs, desc="Processing chunks"):
            chunk_content = doc['content']
            
            if use_context:
                # Get contextual information using Claude
                context = self.situate_context(doc['full_content'], chunk_content)
                
                # Combine original chunk with context for embedding
                if context:
                    contextualized_text = f"{chunk_content}\n\nContext: {context}"
                    doc['context'] = context
                else:
                    contextualized_text = chunk_content
                    doc['context'] = ""
            else:
                contextualized_text = chunk_content
                doc['context'] = ""
            
            # Generate embedding
            try:
                embedding = self.embedding_model.encode([contextualized_text])[0]
                doc['embedding'] = embedding.tolist()
                doc['metadata']['has_context'] = use_context and bool(doc['context'])
                
                # Remove full_content to save space
                doc.pop('full_content', None)
                contextualized_docs.append(doc)
                
            except Exception as e:
                print(f"Warning: Failed to generate embedding for chunk: {e}")
                continue
        
        return contextualized_docs

    def store_and_deploy_vector_db(self, docs: List[Dict[str, Any]], 
                                   index_path: str = "data/vector_db/vector_index.faiss") -> bool:
        """
        Store the enhanced vector database.
        """
        print("Building FAISS index...")
        embeddings = np.array([doc["embedding"] for doc in docs]).astype('float32')
        dimension = embeddings.shape[1]
        
        # Use IndexIVFFlat for better performance with larger datasets
        if len(docs) > 100:
            nlist = min(100, len(docs) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
            index.add(embeddings)
            print(f"  → Created IVF index with {nlist} clusters")
        else:
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            print("  → Created flat L2 index")
        
        # Save index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        
        # Store metadata (without embeddings to save space)
        metadata = []
        for doc in docs:
            metadata_entry = {
                "content": doc["content"],
                "context": doc.get("context", ""),
                **doc["metadata"]
            }
            metadata.append(metadata_entry)
        
        metadata_path = "data/vector_db/vector_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Verify deployment
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise RuntimeError("Vector DB deployment failed: files not created.")
        
        print(f"✅ Enhanced vector DB deployed successfully!")
        print(f"   Index: {index_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Total documents: {len(docs)}")
        print(f"   Embedding dimension: {dimension}")
        print(f"   Documents with context: {sum(1 for doc in docs if doc['metadata'].get('has_context', False))}")
        
        return True


def main(use_contextual: bool = True):
    """Main function to set up enhanced vector database."""
    setup = ContextualVectorSetup()
    
    try:
        # Parse PDFs
        docs = setup.parse_pdfs_to_docs()
        if not docs:
            print("No documents found. Using fallback sample data...")
            from src.util.simple_vector_setup import parse_pdfs_to_docs_simple
            docs = [{"content": d["content"], "full_content": d["content"], "metadata": d["metadata"]} 
                   for d in parse_pdfs_to_docs_simple()]
        
        # Generate embeddings
        enhanced_docs = setup.generate_contextual_embeddings(docs, use_context=use_contextual)
        
        # Store vector database
        success = setup.store_and_deploy_vector_db(enhanced_docs)
        
        if success:
            print("\n🎉 Enhanced PDF parsing and vector DB deployment completed successfully!")
            return True
        
    except Exception as e:
        print(f"❌ Error in enhanced setup: {e}")
        print("Falling back to simple vector setup...")
        
        # Fallback to simple setup
        from src.util.simple_vector_setup import parse_pdfs_to_docs_simple, generate_embeddings, store_and_deploy_vector_db
        docs = parse_pdfs_to_docs_simple()
        docs_with_embeddings = generate_embeddings(docs)
        return store_and_deploy_vector_db(docs_with_embeddings)


if __name__ == "__main__":
    import sys
    use_contextual = "--no-context" not in sys.argv
    main(use_contextual=use_contextual)