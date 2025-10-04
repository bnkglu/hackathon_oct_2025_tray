#!/usr/bin/env python3
"""
Script to set up the vector database from PDF documents.
Run this separately to avoid memory issues.
"""

import os
import sys
sys.path.append('.')

from src.util.pdf_vector_setup import parse_pdfs_to_docs, generate_embeddings, store_and_deploy_vector_db

def main():
    try:
        print('Setting up vector database...')
        print('1. Parsing PDFs...')
        docs = parse_pdfs_to_docs()
        print(f'   Found {len(docs)} document chunks')

        print('2. Generating embeddings...')
        docs_with_embeddings = generate_embeddings(docs)
        print(f'   Generated embeddings for {len(docs_with_embeddings)} chunks')

        print('3. Storing vector database...')
        result = store_and_deploy_vector_db(docs_with_embeddings)
        
        if result:
            print('✅ Vector database setup complete!')
            print('   Files created:')
            print('   - data/vector_db/vector_index.faiss')
            print('   - data/vector_db/vector_metadata.json')
        else:
            print('❌ Vector database setup failed!')
            
    except Exception as e:
        print(f'❌ Error setting up vector database: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()