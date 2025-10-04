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
from sentence_transformers import SentenceTransformer
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
    voyage_api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Add contextual information to each chunk using Claude,
    then generate embeddings with Sentence Transformers (local).

    Uses prompt caching for cost efficiency.
    """
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    # Use Sentence Transformers instead of Voyage AI (runs locally, no rate limits)
    print("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

    BATCH_CHUNK_CONTEXT_PROMPT = """
Here are {num_chunks} chunks from the document. For each chunk, provide a short succinct context to situate it within the overall document for search retrieval purposes.

{chunks_text}

Respond with ONLY a JSON array containing {num_chunks} context strings, one for each chunk in order. Format:
["context for chunk 1", "context for chunk 2", ...]
"""

    def situate_context_batch(doc_content: str, chunks: List[str], doc_summary: str = None):
        """Generate context for multiple chunks in a single Claude request."""
        # For very large documents, use a summary instead of full content
        # Estimate: 1 token ≈ 4 chars, so 200k tokens ≈ 800k chars
        # Use summary if document is > 500k chars to stay safe
        if len(doc_content) > 500000:
            if not doc_summary:
                # Create a brief summary (first 100k chars as context)
                context_text = doc_content[:100000] + "\n\n[Document continues but truncated for context...]"
            else:
                context_text = doc_summary
        else:
            context_text = doc_content

        # Format chunks for prompt
        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            chunks_text += f"\n<chunk{i}>\n{chunk}\n</chunk{i}>\n"

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=context_text),
                    "cache_control": {"type": "ephemeral"}  # Cache the document
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": BATCH_CHUNK_CONTEXT_PROMPT.format(
                        num_chunks=len(chunks),
                        chunks_text=chunks_text
                    ),
                }
            ]
        )

        # Parse JSON response
        import json as json_module
        response_text = response.content[0].text.strip()

        # Debug: Check if response is empty
        if not response_text:
            raise ValueError(f"Empty response from Claude for {len(chunks)} chunks")

        # Try to extract JSON array if response contains additional text
        if not response_text.startswith('['):
            # Find the JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            else:
                # If no JSON array found, log the response and raise error
                print(f"\n⚠️  Warning: Could not find JSON array in response:")
                print(f"Response (first 500 chars): {response_text[:500]}")
                raise ValueError(f"No JSON array found in response")

        try:
            contexts = json_module.loads(response_text)
        except json_module.JSONDecodeError as e:
            print(f"\n⚠️  JSON parsing error:")
            print(f"Response text (first 1000 chars): {response_text[:1000]}")
            raise

        return contexts, response.usage

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
    print("(Using batch processing to respect rate limits)")

    BATCH_SIZE = 5  # Process 5 chunks per API call

    with tqdm(total=total_chunks, desc="Contextualizing") as pbar:
        for doc in docs:
            doc_content = doc["content"]
            chunks = doc["chunks"]

            # Process chunks in batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i+BATCH_SIZE]
                chunk_contents = [c["content"] for c in batch_chunks]

                # Generate contexts for batch with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        contexts, usage = situate_context_batch(doc_content, chunk_contents)
                        break
                    except Exception as e:
                        if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 10  # 10, 20, 30 seconds
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

                # Process each chunk-context pair
                for chunk, context in zip(batch_chunks, contexts):
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

                # Add delay between batches to respect rate limits
                time.sleep(2.0)

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
    print("Generating embeddings with Sentence Transformers...")
    texts_to_embed = [chunk["text_to_embed"] for chunk in all_chunks]

    # Use batch encoding for efficiency
    print(f"Encoding {len(texts_to_embed)} chunks...")
    embeddings = embedding_model.encode(
        texts_to_embed,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

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
    """Main setup function with incremental processing."""
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

        # Define paths
        index_path = "data/vector_db/vector_index.faiss"
        metadata_path = "data/vector_db/vector_metadata.json"
        progress_path = "data/vector_db/processed_pdfs.json"

        # Load progress tracking
        processed_pdfs = set()
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                processed_pdfs = set(json.load(f))
            print(f"📋 Found {len(processed_pdfs)} already processed PDFs\n")

        # Load existing embeddings if they exist
        all_chunks = []
        if os.path.exists(metadata_path) and os.path.exists(index_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)

            # Load existing FAISS index to get embeddings
            import faiss
            index = faiss.read_index(index_path)

            # Reconstruct chunks with embeddings
            for i, metadata in enumerate(existing_metadata):
                embedding = index.reconstruct(i)
                all_chunks.append({
                    "text_to_embed": f"{metadata['original_content']}\n\nContext: {metadata['context']}",
                    "metadata": metadata,
                    "embedding": embedding.tolist()
                })

            print(f"📦 Loaded {len(existing_metadata)} existing chunks from vector DB\n")

        # Step 1: Parse PDFs
        print("Step 1: Parsing PDFs...")
        docs = parse_pdfs_to_docs()

        if not docs:
            print("❌ No documents parsed!")
            return False

        # Step 2: Process each PDF individually
        print("Step 2: Processing PDFs with contextual embeddings...")
        new_chunks_added = 0

        for doc in docs:
            doc_id = doc["doc_id"]

            # Skip if already processed
            if doc_id in processed_pdfs:
                print(f"⏭️  Skipping {doc_id} (already processed)")
                continue

            print(f"\n{'='*70}")
            print(f"Processing: {doc_id}")
            print(f"{'='*70}")

            try:
                # Add contextual embeddings for this document
                chunks_with_embeddings = add_contextual_embeddings(
                    [doc],  # Process one document at a time
                    anthropic_api_key,
                    voyage_api_key
                )

                # Append to all chunks
                all_chunks.extend(chunks_with_embeddings)
                new_chunks_added += len(chunks_with_embeddings)

                # Save progress immediately
                print(f"\n💾 Saving progress for {doc_id}...")
                store_vector_db(all_chunks, index_path, metadata_path)

                # Mark as processed
                processed_pdfs.add(doc_id)
                os.makedirs(os.path.dirname(progress_path), exist_ok=True)
                with open(progress_path, "w") as f:
                    json.dump(list(processed_pdfs), f, indent=2)

                print(f"✅ Completed {doc_id} ({len(chunks_with_embeddings)} chunks)")

            except Exception as e:
                print(f"❌ Error processing {doc_id}: {e}")
                import traceback
                traceback.print_exc()
                print(f"⚠️  Continuing with next document...")
                continue

        # Final summary
        print(f"\n{'='*70}")
        print(f"✅ Processing complete!")
        print(f"{'='*70}")
        print(f"Total processed PDFs: {len(processed_pdfs)}")
        print(f"New chunks added this run: {new_chunks_added}")
        print(f"Total chunks in vector DB: {len(all_chunks)}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
