"""
Tests for DB Agent using questions 40, 41, 42 from public_questions.json.
"""

import pytest
import asyncio
import json
import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

from src.agents.db_agent import DBAgent
from src.agents.aggregator import Aggregator
from src.agents.evaluator import Evaluator
from src.util.client import MCPClient
from src.util.utils import get_root_dir

load_dotenv()


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def db_agent():
    """Initialize DB Agent with MCP client."""
    # Initialize MCP client for database
    db_client = MCPClient()
    await db_client.connect_to_server("python", ["-m", "src.mcp_servers.database"])

    # Initialize Anthropic client
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Create DB Agent
    agent = DBAgent(db_client, anthropic_client)

    yield agent

    # Cleanup
    await db_client.cleanup()


@pytest.fixture(scope="module")
def test_questions():
    """Load test questions from public_questions.json."""
    questions_path = get_root_dir() / "data" / "public_questions.json"
    with open(questions_path, "r") as f:
        all_questions = json.load(f)

    # Return questions 40, 41, 42
    return {
        "40": all_questions["40"],
        "41": all_questions["41"],
        "42": all_questions["42"],
    }


@pytest.mark.asyncio
async def test_question_40_fossil_fuel_diff(db_agent, test_questions):
    """
    Test Q40: What is the absolute difference in fossil fuel consumption
    for the United Kingdom between 2020 and 2015?

    Expected: -383.323 TWh
    """
    question_data = test_questions["40"]
    question = question_data["question"]
    expected_answer = question_data["answer"]
    expected_unit = question_data["unit"]

    # Process question
    response = await db_agent.process(question)

    # Assertions
    assert response.value is not None, "DB Agent returned None"
    assert isinstance(response.value, (int, float)), f"Expected numeric value, got {type(response.value)}"

    # Check answer (with tolerance)
    assert abs(response.value - expected_answer) < 0.01, (
        f"Expected {expected_answer}, got {response.value}"
    )

    # Check unit
    assert response.unit == expected_unit, f"Expected unit {expected_unit}, got {response.unit}"

    # Check sources
    assert len(response.sources) > 0, "No sources returned"
    assert response.sources[0].source_type == "db", "Expected source type 'db'"

    print(f"\n✓ Q40 PASSED: {response.value} {response.unit}")
    print(f"  SQL: {response.metadata.get('sql_query', 'N/A')}")


@pytest.mark.asyncio
async def test_question_41_low_carbon_sum(db_agent, test_questions):
    """
    Test Q41: What was the total low-carbon electricity generated
    by France and Germany combined in 2021?

    Expected: 803.24 TWh
    """
    question_data = test_questions["41"]
    question = question_data["question"]
    expected_answer = question_data["answer"]
    expected_unit = question_data["unit"]

    # Process question
    response = await db_agent.process(question)

    # Assertions
    assert response.value is not None, "DB Agent returned None"
    assert isinstance(response.value, (int, float)), f"Expected numeric value, got {type(response.value)}"

    # Check answer (with tolerance)
    assert abs(response.value - expected_answer) < 0.01, (
        f"Expected {expected_answer}, got {response.value}"
    )

    # Check unit
    assert response.unit == expected_unit, f"Expected unit {expected_unit}, got {response.unit}"

    # Check sources
    assert len(response.sources) > 0, "No sources returned"

    print(f"\n✓ Q41 PASSED: {response.value} {response.unit}")
    print(f"  SQL: {response.metadata.get('sql_query', 'N/A')}")


@pytest.mark.asyncio
async def test_question_42_co2_difference(db_agent, test_questions):
    """
    Test Q42: What is the difference between total CO2 and consumption-based CO2
    for Australia in 2020?

    Expected: 44.05 million tonnes
    """
    question_data = test_questions["42"]
    question = question_data["question"]
    expected_answer = question_data["answer"]
    expected_unit = question_data["unit"]

    # Process question
    response = await db_agent.process(question)

    # Assertions
    assert response.value is not None, "DB Agent returned None"
    assert isinstance(response.value, (int, float)), f"Expected numeric value, got {type(response.value)}"

    # Check answer (with tolerance)
    assert abs(response.value - expected_answer) < 0.01, (
        f"Expected {expected_answer}, got {response.value}"
    )

    # Check unit (allow slight variations)
    assert "million" in response.unit.lower() or response.unit == expected_unit, (
        f"Expected unit containing 'million', got {response.unit}"
    )

    # Check sources
    assert len(response.sources) > 0, "No sources returned"

    print(f"\n✓ Q42 PASSED: {response.value} {response.unit}")
    print(f"  SQL: {response.metadata.get('sql_query', 'N/A')}")


@pytest.mark.asyncio
async def test_end_to_end_pipeline(db_agent, test_questions):
    """
    Test full pipeline: DB Agent → Aggregator → Evaluator
    """
    aggregator = Aggregator()
    evaluator = Evaluator()

    question_data = test_questions["40"]
    question = question_data["question"]

    # Step 1: DB Agent
    agent_response = await db_agent.process(question)

    # Step 2: Aggregator
    aggregated = aggregator.aggregate_single(agent_response, question)

    # Step 3: Evaluator
    final_result = evaluator.evaluate(question, aggregated)

    # Assertions
    assert final_result["question"] == question
    assert final_result["answer"] is not None
    assert final_result["answer_type"] in ["float", "int"]
    assert final_result["sources"] is not None
    assert len(final_result["sources"]) > 0

    print(f"\n✓ END-TO-END PIPELINE PASSED")
    print(f"  Final result: {json.dumps(final_result, indent=2)}")
