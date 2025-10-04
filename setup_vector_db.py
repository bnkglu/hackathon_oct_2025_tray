#!/usr/bin/env python3
"""
Script to set up the vector database from PDF documents using contextual embeddings.

Run this to create the vector database for RAG.
"""

import os
import sys
sys.path.append('.')

from dotenv import load_dotenv
from src.util.pdf_vector_setup import main

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Set Voyage API key if not already set
    if not os.getenv("VOYAGE_API_KEY"):
        # Use the provided API key
        os.environ["VOYAGE_API_KEY"] = "pa-zevF3GXDcowQ_ww8rxHU0WFiUj1wZ943_efWnzQ6nSI"

    success = main()
    sys.exit(0 if success else 1)
