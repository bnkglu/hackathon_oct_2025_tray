"""
This script acts as the MCP host, the AI agent which calls tools to answer questions.

Most of the script was copied and adapted from the official MCP documentation "Build an MCP client" available at:
https://modelcontextprotocol.io/docs/develop/build-client
raw file available here:
https://github.com/modelcontextprotocol/quickstart-resources/blob/main/mcp-client-python/client.py
"""

import asyncio
from src.util.client import MCPClient

from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class Agent:
    def __init__(self):
        self.tools = []
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) # Initialise Claude

    async def initialise_servers(self):

        # Initialize tools lists
        self.wikipedia_tools = []
        self.database_tools = []

        # Initialise the Wikipedia MCP server
        try:
            self.wikipedia_client = MCPClient()
            await self.wikipedia_client.connect_to_server("python", ["-m", "wikipedia_mcp", "--transport", "stdio"])

            wikipedia_tools = await self.wikipedia_client.list_tools()
            self.wikipedia_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in wikipedia_tools]
            print("Wikipedia MCP server initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Wikipedia MCP server: {e}")
            # Ensure client is properly cleaned up if initialization fails
            if hasattr(self, 'wikipedia_client') and self.wikipedia_client is not None:
                try:
                    await self.wikipedia_client.cleanup()
                except:
                    pass
            self.wikipedia_client = None

        # Initialise the OWID database MCP server
        try:
            self.database_client = MCPClient()
            await self.database_client.connect_to_server("python", ["-m", "src.mcp_servers.remote_database"])

            database_tools = await self.database_client.list_tools()
            self.database_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in database_tools]
            print("Database MCP server initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Database MCP server: {e}")
            # Ensure client is properly cleaned up if initialization fails
            if hasattr(self, 'database_client') and self.database_client is not None:
                try:
                    await self.database_client.cleanup()
                except:
                    pass
            self.database_client = None

        # TODO: add other MCP servers here as needed

        self.tools = self.wikipedia_tools + self.database_tools + []  # TODO: put all tools here

    async def answer_question(self, question: str) -> str:
        """
        Answer a question by calling tools as needed.
        :param question: a single question as a string
        :return: the answer as a string
        """
        messages = ...

        # Get a response from Claude like this
        response = self.anthropic.messages.create(
            model="claude-3-haiku-20240307", # TODO: change as needed (check available_anthropic_models.json)
            max_tokens=1000, # TODO: change as needed
            messages=messages,
            tools=self.tools
        )

        # TODO: IMPORTANT: if you put Claude calls in a while loop, make sure to enforce MAX ITERATIONS

        # TODO: parse the response and call tools as needed

        return 'This should be an answer.' # TODO: return the actual answer to the question


async def main(verbose: bool = True):
    agent = Agent()
    questions = ... # TODO: get questions from public_questions.json or private_questions.json
    answers = []
    try:
        await agent.initialise_servers()
        # TODO: change this loop as needed
        for i, q in enumerate(questions, 1):
            if verbose:
                print(f"Answering question {i}/{len(questions)}...")
            try:
                answer = await agent.answer_question(q)
                answers.append(answer)
                if verbose:
                    print(f"{i}.\nQuestion: {q}\nAnswer: {answer}\n")
            except Exception as e:
                print(f"Error answering question {i}: {e}")
                answers.append(None)  # keep placeholder so indexes match
    finally:
        if hasattr(agent, 'wikipedia_client') and agent.wikipedia_client is not None:
            try:
                await agent.wikipedia_client.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up Wikipedia client: {e}")
            finally:
                agent.wikipedia_client = None

        if hasattr(agent, 'database_client') and agent.database_client is not None:
            try:
                await agent.database_client.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up Database client: {e}")
            finally:
                agent.database_client = None
        # Cleanup other clients as needed here
    # TODO: save answers in json file and/or send directly to central server
    return answers

if __name__ == "__main__":
    answers = asyncio.run(main())
