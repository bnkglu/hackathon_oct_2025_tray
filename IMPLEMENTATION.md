# Multi-Agent QA System Implementation

## Overview

This document describes the implemented multi-agent system for answering sustainability questions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Question                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Router Agent                            │
│  (Classifies: db / wikipedia / rag / hybrid)                │
└─────────────┬───────────────────────────────────────────────┘
              │
              ├──► DB Agent ───┐
              │                 │
              ├──► Wikipedia ───┤
              │    (future)     │
              │                 ├──► Aggregator ──► Evaluator ──► Final Answer
              └──► RAG Agent ───┤
                   (future)     │
                                │
                   ────────────┘
```

### Components

1. **Base Infrastructure** (`src/agents/base.py`)
   - `BaseAgent`: Abstract base class for all agents
   - `AgentResponse`: Standardized response format
   - `SourceInfo`: Source citation structure

2. **Router Agent** (`src/agents/router.py`)
   - Classifies questions using Claude Haiku 3.5
   - Routes to: `db`, `wikipedia`, `rag`, or `hybrid`
   - Returns routing decision with confidence and reasoning

3. **DB Agent** (`src/agents/db_agent.py`)
   - Queries SQL database via MCP server
   - Uses Claude Sonnet 4.5 for SQL generation
   - Two-step process:
     1. Get table schemas from MCP resources
     2. Generate and execute SQL query
   - Handles results from `owid_energy_data` and `owid_co2_data` tables

4. **Aggregator** (`src/agents/aggregator.py`)
   - Combines results from multiple agents
   - Performs arithmetic: sum, difference, ratio, average
   - Merges sources and maintains reasoning trace

5. **Evaluator** (`src/agents/evaluator.py`)
   - Final quality control
   - Validates answer types (float/int/string/bool/list)
   - Applies rounding (3 decimal places for floats)
   - Detects hallucinations (e.g., "Scope 5")
   - Formats to submission schema

6. **Main Orchestrator** (`src/agent.py`)
   - Initializes MCP servers and agents
   - Runs full pipeline: Router → Agent → Aggregator → Evaluator
   - Handles cleanup and error recovery

## Usage

### Running Tests

Test with questions 40, 41, 42:

```bash
pytest tests/test_db_agent.py -v
```

### Running the Agent

Process first 3 questions (default):
```bash
python -m src.agent
```

Process specific number of questions:
```bash
python -m src.agent 10  # Process first 10 questions
```

Process all questions:
```bash
python -m src.agent 0  # 0 = all questions
```

### Evaluation Script

Evaluate DB Agent on database questions:

```bash
# Evaluate on first 10 DB questions
python -m src.evaluate

# Evaluate on specific number
python -m src.evaluate 20

# Evaluate on all DB questions
python -m src.evaluate 0
```

## Files Created

```
src/
├── agents/
│   ├── __init__.py           # Module init
│   ├── base.py               # BaseAgent, AgentResponse, SourceInfo
│   ├── router.py             # Router Agent (question classification)
│   ├── db_agent.py           # DB Agent (SQL queries)
│   ├── aggregator.py         # Result aggregation and arithmetic
│   └── evaluator.py          # Final validation and formatting
├── agent.py                  # Main orchestrator (UPDATED)
└── evaluate.py               # Evaluation script

tests/
├── __init__.py
└── test_db_agent.py          # Tests for Q40-42
```

## Current Status

### ✅ Implemented
- Base agent infrastructure
- Router Agent (question classification)
- DB Agent (SQL generation and execution)
- Aggregator (result combination)
- Evaluator (validation and formatting)
- Test suite for questions 40-42
- Evaluation script for DB questions
- Main pipeline integration

### 🚧 TODO (Future Work)
- Wikipedia Agent (use existing wikipedia_mcp server)
- RAG Agent (PDF document retrieval with embeddings)
- Hybrid multi-agent orchestration
- Currency converter MCP server implementation
- Advanced error recovery and retry logic

## Test Questions

The implementation was designed to handle these test questions:

**Q40**: What is the absolute difference in fossil fuel consumption for the United Kingdom between 2020 and 2015?
- Expected: -383.323 TWh

**Q41**: What was the total low-carbon electricity generated by France and Germany combined in 2021?
- Expected: 803.24 TWh

**Q42**: What is the difference between total CO2 and consumption-based CO2 for Australia in 2020?
- Expected: 44.05 million tonnes

## Key Design Decisions

1. **Model Selection**
   - Sonnet 4.5 for SQL generation (high accuracy)
   - Haiku 3.5 for routing (speed/cost efficiency)

2. **Structured Responses**
   - All agents return `AgentResponse` objects
   - Standardized intermediate format enables composition

3. **Traceability**
   - Every response includes sources and metadata
   - SQL queries logged for debugging
   - Reasoning traces maintained throughout pipeline

4. **Safety**
   - MAX_ITERATIONS enforced in main loop
   - Hallucination detection in Evaluator
   - Graceful error handling and cleanup

5. **Incremental Testing**
   - Start with 3 test questions (Q40-42)
   - Expand to all 43 DB questions
   - Then add Wikipedia and RAG agents
