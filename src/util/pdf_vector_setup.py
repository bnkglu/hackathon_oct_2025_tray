"""
PDF to Vector DB setup using pypdf + Contextual Embeddings.

Based on Anthropic's Contextual Retrieval guide:
https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings
"""

import os
import json
import faiss
import numpy as np
from pypdf import PdfReader
from anthropic import Anthropic
import voyageai
from typing import List, Dict, Any
from tqdm import tqdm
import threading
import time


def parse_pdfs_to_docs(data_dir: str = "data/annual_reports") -> List[Dict[str, Any]]:
    """
    Parse PDFs using pypdf and split into chunks.

    Returns list of documents with chunks.
    """
    print(f"Parsing PDFs from {data_dir}...")

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files\n")

    docs = []

    for idx, filename in enumerate(pdf_files, 1):
        try:
            path = os.path.join(data_dir, filename)
            print(f"[{idx}/{len(pdf_files)}] Processing {filename}...")

            # Read PDF
            reader = PdfReader(path)
            total_pages = len(reader.pages)
            print(f"  Pages: {total_pages}")

            # Extract full document text
            full_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():
                    full_text.append(text)

            doc_content = "\n\n".join(full_text)
            print(f"  ✓ Extracted {len(doc_content)} chars")

            # Split into chunks (by paragraphs)
            paragraphs = doc_content.split('\n\n')
            chunks = []
            current_chunk = ""
            chunk_id = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If adding this paragraph exceeds 1000 chars, save current chunk
                if len(current_chunk) + len(para) > 1000 and current_chunk:
                    if len(current_chunk) > 100:  # Only keep substantial chunks
                        chunks.append({
                            "content": current_chunk,
                            "chunk_id": chunk_id
                        })
                        chunk_id += 1
                    current_chunk = para
                else:
                    current_chunk = current_chunk + "\n\n" + para if current_chunk else para

            # Add last chunk
            if current_chunk and len(current_chunk) > 100:
                chunks.append({
                    "content": current_chunk,
                    "chunk_id": chunk_id
                })

            print(f"  ✓ Created {len(chunks)} chunks")

            # Create document structure
            docs.append({
                "doc_id": filename,
                "content": doc_content,
                "chunks": chunks
            })

            print(f"  ✓ Added {len(chunks)} chunks from {filename}\n")

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_chunks = sum(len(doc["chunks"]) for doc in docs)
    print(f"✅ Total: {len(docs)} documents, {total_chunks} chunks\n")
    return docs


def add_contextual_embeddings(
    docs: List[Dict[str, Any]],
    anthropic_api_key: str,
    voyage_api_key: str
) -> List[Dict[str, Any]]:
    """
    Add contextual information to each chunk using Claude,
    then generate embeddings with Voyage AI.

    Uses prompt caching for cost efficiency.
    """
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    voyage_client = voyageai.Client(api_key=voyage_api_key)

    DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

    CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

    def situate_context(doc_content: str, chunk_content: str):
        """Generate context for a chunk using Claude with prompt caching."""
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_content),
                    "cache_control": {"type": "ephemeral"}  # Cache the document
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content),
                }
            ]
        )
        return response.content[0].text, response.usage

    # Track token usage
    token_counts = {
        'input': 0,
        'output': 0,
        'cache_read': 0,
        'cache_creation': 0
    }
    token_lock = threading.Lock()

    # Process each document
    all_chunks = []
    total_chunks = sum(len(doc["chunks"]) for doc in docs)

    print("Generating contextual information for chunks...")
    print("(Adding delays to respect rate limits)")

    with tqdm(total=total_chunks, desc="Contextualizing") as pbar:
        for doc in docs:
            doc_content = doc["content"]

            for chunk in doc["chunks"]:
                # Generate context with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        context, usage = situate_context(doc_content, chunk["content"])
                        break
                    except Exception as e:
                        if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                            pbar.write(f"Rate limit hit, waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise

                # Track tokens
                with token_lock:
                    token_counts['input'] += usage.input_tokens
                    token_counts['output'] += usage.output_tokens
                    token_counts['cache_read'] += usage.cache_read_input_tokens
                    token_counts['cache_creation'] += usage.cache_creation_input_tokens

                # Combine chunk with context
                contextualized_text = f"{chunk['content']}\n\nContext: {context}"

                all_chunks.append({
                    "text_to_embed": contextualized_text,
                    "metadata": {
                        "source_name": doc["doc_id"],
                        "source_type": "pdf",
                        "chunk_id": chunk["chunk_id"],
                        "original_content": chunk["content"],
                        "context": context
                    }
                })

                pbar.update(1)

                # Add delay to respect rate limits (0.1s = max 10 requests/sec)
                time.sleep(0.1)

    # Print token usage stats
    print(f"\n📊 Token Usage Statistics:")
    print(f"  Input tokens (without caching): {token_counts['input']:,}")
    print(f"  Output tokens: {token_counts['output']:,}")
    print(f"  Cache creation tokens: {token_counts['cache_creation']:,}")
    print(f"  Cache read tokens: {token_counts['cache_read']:,}")

    total_tokens = token_counts['input'] + token_counts['cache_read'] + token_counts['cache_creation']
    if total_tokens > 0:
        savings = (token_counts['cache_read'] / total_tokens) * 100
        print(f"  💰 Cache savings: {savings:.1f}% of tokens read from cache (90% discount!)\n")

    # Generate embeddings
    print("Generating embeddings with Voyage AI...")
    texts_to_embed = [chunk["text_to_embed"] for chunk in all_chunks]

    batch_size = 128
    embeddings = []

    with tqdm(total=len(texts_to_embed), desc="Embedding") as pbar:
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            batch_embeddings = voyage_client.embed(
                batch,
                model="voyage-2"
            ).embeddings
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    # Add embeddings to chunks
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk["embedding"] = embedding

    print(f"✅ Generated {len(embeddings)} embeddings\n")
    return all_chunks


def store_vector_db(
    chunks: List[Dict[str, Any]],
    index_path: str = "data/vector_db/vector_index.faiss",
    metadata_path: str = "data/vector_db/vector_metadata.json"
) -> bool:
    """Store embeddings in FAISS index and metadata in JSON."""

    print("Creating FAISS vector database...")

    # Extract embeddings
    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype('float32')
    dimension = embeddings.shape[1]

    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {len(embeddings)}")

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Create output directory
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Save index
    faiss.write_index(index, index_path)
    print(f"  ✓ Index saved: {index_path}")

    # Save metadata
    metadata = [chunk["metadata"] for chunk in chunks]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_path} ({len(metadata)} chunks)")

    print(f"\n✅ Vector database created successfully!")
    return True


def main():
    """Main setup function."""
    try:
        print("=" * 70)
        print("PDF to Vector DB Setup (Contextual Embeddings)")
        print("=" * 70)
        print()

        # Get API keys from environment
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        voyage_api_key = os.getenv("VOYAGE_API_KEY")

        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        if not voyage_api_key:
            raise ValueError("VOYAGE_API_KEY not set in environment")

        # Step 1: Parse PDFs
        print("Step 1: Parsing PDFs...")
        docs = parse_pdfs_to_docs()

        if not docs:
            print("❌ No documents parsed!")
            return False

        # Step 2: Add contextual embeddings
        print("Step 2: Adding contextual embeddings...")
        chunks_with_embeddings = add_contextual_embeddings(
            docs,
            anthropic_api_key,
            voyage_api_key
        )

        # Step 3: Store vector database
        print("Step 3: Storing vector database...")
        success = store_vector_db(chunks_with_embeddings)

        return success

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
