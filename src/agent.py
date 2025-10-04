"""
Multi-agent QA system for sustainability questions.

Architecture:
- Router Agent: Classifies questions and routes to appropriate agents
- DB Agent: Handles SQL database queries
- Wikipedia Agent: Handles encyclopedia queries (future)
- RAG Agent: Handles PDF/document queries (future)
- Aggregator: Combines results from multiple agents
- Evaluator: Final validation and formatting
"""

import asyncio
import json
import logging
from pathlib import Path
from src.util.client import MCPClient
from src.util.utils import get_root_dir
from anthropic import Anthropic
from dotenv import load_dotenv
import os

from src.agents.router import RouterAgent
from src.agents.db_agent import DBAgent
from src.agents.rag_agent import RAGAgent
from src.agents.aggregator import Aggregator
from src.agents.evaluator import Evaluator

load_dotenv()
logging.basicConfig(level=logging.INFO)


class Agent:
    """
    Main orchestrator for the multi-agent QA system.

    Manages initialization of MCP servers and specialized agents,
    routes questions, and coordinates the pipeline.
    """

    def __init__(self):
        """Initialize the agent with Anthropic client."""
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Specialized agents (initialized later)
        self.router = None
        self.db_agent = None
        self.rag_agent = None
        self.aggregator = Aggregator()
        self.evaluator = Evaluator()

        # MCP clients
        self.database_client = None
        self.wikipedia_client = None
        self.vector_client = None

    async def initialise_servers(self):
        """
        Initialize MCP servers and specialized agents.

        Connects to:
        - Database MCP server (src.mcp_servers.database)
        - Wikipedia MCP server (optional, for future use)
        """
        logging.info("Initializing MCP servers and agents...")

        # Initialize Database MCP server
        try:
            self.database_client = MCPClient()
            await self.database_client.connect_to_server(
                "python", ["-m", "src.mcp_servers.database"]
            )
            logging.info("✓ Database MCP server initialized")

            # Initialize DB Agent
            self.db_agent = DBAgent(self.database_client, self.anthropic)
            logging.info("✓ DB Agent initialized")

        except Exception as e:
            logging.error(f"✗ Failed to initialize Database MCP server: {e}")
            if self.database_client is not None:
                try:
                    await self.database_client.cleanup()
                except:
                    pass
            self.database_client = None
            self.db_agent = None

        # Initialize Wikipedia MCP server (optional)
        try:
            self.wikipedia_client = MCPClient()
            await self.wikipedia_client.connect_to_server(
                "python", ["-m", "wikipedia_mcp", "--transport", "stdio"]
            )
            logging.info("✓ Wikipedia MCP server initialized")
        except Exception as e:
            logging.warning(f"Wikipedia MCP server not available: {e}")
            if self.wikipedia_client is not None:
                try:
                    await self.wikipedia_client.cleanup()
                except:
                    pass
            self.wikipedia_client = None

        # Initialize Vector MCP server for RAG
        try:
            self.vector_client = MCPClient()
            await self.vector_client.connect_to_server(
                "python", ["-m", "src.mcp_servers.vector_server"]
            )
            logging.info("✓ Vector MCP server initialized")
            
            # Initialize RAG Agent
            self.rag_agent = RAGAgent(self.vector_client, self.anthropic)
            logging.info("✓ RAG Agent initialized")
        except Exception as e:
            logging.warning(f"Vector MCP server not available: {e}")
            if self.vector_client is not None:
                try:
                    await self.vector_client.cleanup()
                except:
                    pass
            self.vector_client = None
            self.rag_agent = None

        # Initialize Router Agent
        self.router = RouterAgent(self.anthropic)
        logging.info("✓ Router Agent initialized")

        logging.info("All agents initialized successfully\n")

    async def answer_question(self, question: str) -> dict:
        """
        Answer a question using the multi-agent pipeline.

        Pipeline:
        1. Router classifies the question
        2. Appropriate agent(s) process the question
        3. Aggregator combines results (if multiple)
        4. Evaluator validates and formats final answer

        Parameters
        ----------
        question : str
            Natural language question

        Returns
        -------
        dict
            Final formatted answer with question, answer, sources, etc.
        """
        MAX_ITERATIONS = 3  # Safety limit for any loops
        iteration = 0

        try:
            # Step 1: Route the question
            routing = await self.router.route(question)
            logging.info(
                f"Router decision: {routing.route_type} "
                f"(confidence: {routing.confidence:.2f})"
            )
            logging.info(f"Reasoning: {routing.reasoning}")

            # Step 2: Process based on routing
            if routing.route_type == "db" and self.db_agent:
                # Use DB Agent
                agent_response = await self.db_agent.process(question)
                aggregated = self.aggregator.aggregate_single(agent_response, question)

            elif routing.route_type == "wikipedia":
                # TODO: Implement Wikipedia agent
                logging.warning("Wikipedia agent not yet implemented")
                aggregated = {
                    "value": None,
                    "unit": None,
                    "sources": [],
                    "comment": "Wikipedia agent not yet implemented",
                }

            elif routing.route_type == "rag" and self.rag_agent:
                # Use RAG Agent
                agent_response = await self.rag_agent.process(question)
                aggregated = self.aggregator.aggregate_single(agent_response, question)
            
            elif routing.route_type == "rag":
                # RAG agent not available
                logging.warning("RAG agent not initialized - vector database may not be available")
                aggregated = {
                    "value": None,
                    "unit": None,
                    "sources": [],
                    "comment": "RAG agent not available - please ensure vector database is set up",
                }

            elif routing.route_type == "hybrid":
                # TODO: Implement hybrid multi-agent processing
                logging.warning("Hybrid routing not yet implemented, defaulting to DB")
                if self.db_agent:
                    agent_response = await self.db_agent.process(question)
                    aggregated = self.aggregator.aggregate_single(agent_response, question)
                else:
                    aggregated = {
                        "value": None,
                        "unit": None,
                        "sources": [],
                        "comment": "No agent available",
                    }

            else:
                aggregated = {
                    "value": None,
                    "unit": None,
                    "sources": [],
                    "comment": f"Unknown route type: {routing.route_type}",
                }

            # Step 3: Evaluate and format final answer
            final_answer = self.evaluator.evaluate(question, aggregated)

            return final_answer

        except Exception as e:
            logging.error(f"Error in answer_question: {e}", exc_info=True)
            return {
                "question": question,
                "answer": None,
                "answer_type": "float",
                "unit": None,
                "difficulty": "unknown",
                "comment": f"Error: {str(e)}",
                "sources": None,
            }


async def main(
    questions_file: str = "public_questions.json",
    output_file: str = "submission.json",
    limit: int = None,
    verbose: bool = True,
):
    """
    Main entry point for answering questions.

    Parameters
    ----------
    questions_file : str
        Name of questions file in data/ directory (default: "public_questions.json")
    output_file : str
        Output file name for submission (default: "submission.json")
    limit : int
        Limit number of questions to process (for testing)
    verbose : bool
        Print progress information
    """
    # Load questions
    questions_path = get_root_dir() / "data" / questions_file
    with open(questions_path, "r") as f:
        all_questions = json.load(f)

    # Filter to DB questions only for now (or use limit)
    if limit:
        question_ids = list(all_questions.keys())[:limit]
    else:
        # Process all questions
        question_ids = list(all_questions.keys())

    logging.info(f"Loaded {len(question_ids)} questions from {questions_file}")

    # Initialize agent
    agent = Agent()
    results = {}

    try:
        await agent.initialise_servers()

        # Process each question
        for i, q_id in enumerate(question_ids, 1):
            question_text = all_questions[q_id]["question"]

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Question {i}/{len(question_ids)} (ID: {q_id})")
                print(f"{'=' * 70}")
                print(f"Q: {question_text}\n")

            try:
                answer_dict = await agent.answer_question(question_text)
                results[q_id] = answer_dict

                if verbose:
                    print(f"A: {answer_dict['answer']} {answer_dict.get('unit', '')}")
                    print(f"Type: {answer_dict['answer_type']}")
                    print(f"Sources: {answer_dict.get('sources', [])}")
                    print(f"Comment: {answer_dict.get('comment', '')[:100]}...")

            except Exception as e:
                logging.error(f"Error answering question {q_id}: {e}", exc_info=True)
                results[q_id] = {
                    "question": question_text,
                    "answer": None,
                    "answer_type": "float",
                    "unit": None,
                    "difficulty": "unknown",
                    "comment": f"Error: {str(e)}",
                    "sources": None,
                }

    finally:
        # Cleanup MCP clients
        if agent.wikipedia_client is not None:
            try:
                await agent.wikipedia_client.cleanup()
            except Exception as e:
                logging.warning(f"Error cleaning up Wikipedia client: {e}")
            agent.wikipedia_client = None

        if agent.database_client is not None:
            try:
                await agent.database_client.cleanup()
            except Exception as e:
                logging.warning(f"Error cleaning up Database client: {e}")
            agent.database_client = None
            
        if agent.vector_client is not None:
            try:
                await agent.vector_client.cleanup()
            except Exception as e:
                logging.warning(f"Error cleaning up Vector client: {e}")
            agent.vector_client = None

    # Save results
    output_path = get_root_dir() / output_file
    submission = {"team_name": "tray", "answers": results}

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    logging.info(f"\nResults saved to {output_path}")
    logging.info(f"Answered {len(results)} questions")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    # Run with first 3 questions for testing
    if limit is None:
        limit = 3  # Default to first 3 for initial testing

    asyncio.run(main(limit=limit, verbose=True))
