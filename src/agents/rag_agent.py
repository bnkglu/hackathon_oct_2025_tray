"""
RAG Agent for document-based question answering using vector search.
"""

import json
import logging
from typing import List, Dict, Any
from anthropic import Anthropic

from .base import BaseAgent, AgentResponse, SourceInfo
from ..util.client import MCPClient

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    RAG (Retrieval-Augmented Generation) agent for PDF document queries.
    
    Uses vector search to find relevant document chunks, then uses Claude
    to extract specific information and answer questions.
    """

    def __init__(self, vector_client: MCPClient, anthropic: Anthropic):
        """
        Initialize RAG agent.
        
        Parameters
        ----------
        vector_client : MCPClient
            Connected MCP client for vector search
        anthropic : Anthropic
            Anthropic client for Claude API
        """
        super().__init__("RAGAgent")
        self.vector_client = vector_client
        self.anthropic = anthropic

    async def process(self, question: str) -> AgentResponse:
        """
        Process a question using RAG approach.
        
        Steps:
        1. Search for relevant documents using vector similarity
        2. Use Claude to extract specific information from retrieved documents
        3. Return structured response
        
        Parameters
        ----------
        question : str
            Natural language question
            
        Returns
        -------
        AgentResponse
            Structured response with extracted information
        """
        try:
            # Step 1: Vector search for relevant documents
            search_results = await self._search_documents(question)
            
            if not search_results or not search_results.get("results"):
                return AgentResponse(
                    value=None,
                    unit=None,
                    confidence=0.0,
                    metadata={"error": "No relevant documents found"}
                )
            
            # Step 2: Use Claude to extract information
            claude_response = await self._extract_with_claude(question, search_results)
            
            # Step 3: Parse and structure the response
            return self._structure_response(claude_response, search_results)
            
        except Exception as e:
            logger.error(f"Error in RAG agent processing: {e}", exc_info=True)
            return AgentResponse(
                value=None,
                unit=None,
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def _search_documents(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Search for relevant documents using vector similarity.
        
        Parameters
        ----------
        question : str
            Search query
        k : int
            Number of documents to retrieve
            
        Returns
        -------
        Dict[str, Any]
            Search results from vector database
        """
        try:
            # Use the MCP client to call the vector search tool
            result = await self.vector_client.call_tool(
                "vector_search",
                {"query": question, "k": k}
            )
            
            if result.isError:
                logger.error(f"Vector search error: {result.content}")
                return {}
                
            # The result should be JSON data directly
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    # Parse JSON response from vector search
                    search_data = json.loads(result.content[0].text)
                else:
                    # Direct content
                    search_data = json.loads(result.content)
            else:
                logger.error("Empty result from vector search")
                return {}
                
            logger.info(f"Found {search_data.get('num_results', 0)} relevant documents")
            return search_data
            
        except Exception as e:
            logger.error(f"Error in document search: {e}")
            return {}

    async def _extract_with_claude(self, question: str, search_results: Dict[str, Any]) -> str:
        """
        Use Claude to extract specific information from retrieved documents.
        
        Parameters
        ----------
        question : str
            Original question
        search_results : Dict[str, Any]
            Results from vector search
            
        Returns
        -------
        str
            Claude's response with extracted information
        """
        # Prepare context from search results
        context_pieces = []
        for i, result in enumerate(search_results.get("results", []), 1):
            content = result.get("content", "")
            source = result.get("source_name", "unknown")
            similarity = result.get("similarity_score", 0)
            
            context_pieces.append(
                f"Document {i} (Source: {source}, Relevance: {similarity:.3f}):\n{content}\n"
            )
        
        context = "\n".join(context_pieces)
        
        # Create prompt for information extraction
        prompt = f"""You are analyzing sustainability and ESG documents to answer specific questions. 

Based on the following document excerpts, please answer this question: "{question}"

Document excerpts:
{context}

Instructions:
1. If you can find a specific numerical value, extract it precisely
2. Include the unit of measurement if available
3. If no specific number is available, provide the most relevant qualitative information
4. Be precise and cite which document(s) you're using
5. If the information is not available in the documents, say so clearly

Your response should be structured as:
VALUE: [the specific value or "Not specified"]
UNIT: [unit of measurement or "Not specified"] 
CONFIDENCE: [high/medium/low based on how clear the information is]
EXPLANATION: [brief explanation of what you found and from which source]

Answer:"""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error calling Claude: {e}")
            return f"Error extracting information: {str(e)}"

    def _structure_response(self, claude_response: str, search_results: Dict[str, Any]) -> AgentResponse:
        """
        Structure Claude's response into AgentResponse format.
        
        Parameters
        ----------
        claude_response : str
            Claude's structured response
        search_results : Dict[str, Any]
            Original search results for source information
            
        Returns
        -------
        AgentResponse
            Structured response
        """
        try:
            # Parse Claude's structured response
            lines = claude_response.strip().split('\n')
            value = None
            unit = None
            confidence_str = "medium"
            explanation = ""
            
            for line in lines:
                if line.startswith("VALUE:"):
                    value_str = line.replace("VALUE:", "").strip()
                    if value_str.lower() not in ["not specified", "not available", "unknown"]:
                        # Try to parse as number
                        try:
                            value = float(value_str.replace(",", ""))
                        except ValueError:
                            value = value_str
                            
                elif line.startswith("UNIT:"):
                    unit_str = line.replace("UNIT:", "").strip()
                    if unit_str.lower() not in ["not specified", "not available", "unknown"]:
                        unit = unit_str
                        
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.replace("CONFIDENCE:", "").strip().lower()
                    
                elif line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()
            
            # Convert confidence to numeric
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            confidence = confidence_map.get(confidence_str, 0.7)
            
            # Extract source information
            sources = []
            for result in search_results.get("results", []):
                sources.append(SourceInfo(
                    source_name=result.get("source_name", "unknown"),
                    source_type="pdf",
                    page_number=result.get("chunk_number")
                ))
            
            return AgentResponse(
                value=value,
                unit=unit,
                sources=sources,
                confidence=confidence,
                metadata={
                    "search_query": search_results.get("query", ""),
                    "num_documents_found": search_results.get("num_results", 0),
                    "claude_explanation": explanation,
                    "claude_response": claude_response
                }
            )
            
        except Exception as e:
            logger.error(f"Error structuring response: {e}")
            return AgentResponse(
                value=None,
                unit=None,
                confidence=0.0,
                metadata={"error": f"Failed to structure response: {str(e)}"}
            )