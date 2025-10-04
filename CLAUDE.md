# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AIM Hackathon (October 2025)** project for building an agentic AI system to answer sustainability questions using:
- **Claude API** (Anthropic) as the reasoning engine
- **Model Context Protocol (MCP)** for tool integration
- Remote **SQL Server database** (Our World in Data)
- **Wikipedia MCP server** for knowledge retrieval
- **Currency converter** MCP server (custom)
- **PDF annual reports** (in `data/annual_reports/`)

The goal is to answer questions from `data/public_questions.json` and format responses per `data/sample_submission.json`, then submit to the leaderboard at https://hackathon-server.ashysand-de33d6c5.westeurope.azurecontainerapps.io/

## Commands

- **Install dependencies**: `uv sync`
- **Add new package**: `uv add <package-name>`
- **Run the agent**: `python -m src.agent` (after implementing TODOs)
- **Lint**: `flake8`, `ruff check`
- **Format**: `black .`, `ruff format`
- **Type check**: `mypy`

## Architecture

### Core Components

1. **Agent (`src/agent.py`)**
   - Main entry point: `Agent` class orchestrates question answering
   - `initialise_servers()`: Connects to MCP servers (Wikipedia, database, currency converter)
   - `answer_question(question: str) -> str`: Implements agentic loop to answer questions using Claude + tools
   - `main()`: Loads questions from JSON, runs agent on each, saves answers
   - **CRITICAL**: Enforce MAX_ITERATIONS in any while loop to prevent infinite loops

2. **MCP Client (`src/util/client.py`)**
   - `MCPClient` class wraps MCP session management
   - `connect_to_server(command, server_args)`: Connects to an MCP server via stdio
   - `call_tool(tool_name, tool_args)`: Executes tool calls
   - Always set `cwd` to project root via `get_root_dir()` to avoid import errors

3. **MCP Servers (`src/mcp_servers/`)**
   - **`database.py`**: Remote SQL Server MCP server using `pymssql`
     - Tool: `query_database(sql_query)` - Execute SQL queries on Our World in Data database
     - Resource: `schema://{table_name}` - Get table schema
     - Resource: `tables://` - List all tables
   - **`currency_converter.py`**: Currency conversion MCP server (INCOMPLETE - needs implementation)
     - Should use `data/currencies/currency_rates.json` (rates from Sept 26, 2025)
     - Should use `data/currencies/currency_names.json` for currency names

4. **Data Files**
   - `data/public_questions.json`: Questions with answers for development/testing
   - `data/sample_submission.json`: Format for submitting answers (see structure below)
   - `data/annual_reports/*.pdf`: PDF reports (Erste Group, GSK, Swisscom)
   - `data/currencies/*.json`: Currency conversion data

### Answer Submission Format

```json
{
  "team_name": "your-team-name",
  "answers": {
    "1": {
      "question": "...",
      "answer": 42,  // Can be float, int, bool, string, or list
      "sources": [{"source_name": "...", "source_type": "PDF", "page_number": 123}]
    }
  }
}
```

### MCP Server Integration Pattern

When adding a new MCP server to the agent:
1. Create `MCPClient()` instance
2. Call `await client.connect_to_server("python", ["-m", "module.path"])`
3. Extract tools via `await client.list_tools()`
4. Convert to Anthropic tool format: `{"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}`
5. Add to `self.tools` list
6. Add cleanup in `finally` block of `main()`

### Environment Variables

Required in `.env` (see `.env.template`):
- `ANTHROPIC_API_KEY`: Your Claude API key
- `SQLSERVER_HOST`, `SQLSERVER_PORT`, `SQLSERVER_USER`, `SQLSERVER_PASSWORD`, `SQLSERVER_DB`: Database credentials (pre-configured in template)

## Code Style

- **KISS**: KEEP IT SIMPLE STUPID. Do not over-engineer solutions.
- **Python version**: 3.13+ (as per `pyproject.toml`)
- **Line length**: 100 characters max
- **Formatter**: black with isort (profile=black) or ruff
- **Type hints**: Always use type annotations for all parameters and return types
- **Imports**: Use absolute imports at top of file (never inside functions), organize with isort (profile=black)
- **Error handling**: Use specific exception types with logging (avoid print statements)
- **Naming**: `snake_case` for variables/functions, `CamelCase` for classes
- **Whitespaces**: No trailing whitespaces, 4 spaces indentation, no whitespace on blank lines
- **Blank lines**: Never indent blank lines (indent = 0 spaces)
- **Strings**: Use double quotes, f-strings for interpolation
- **Docstrings**: Numpy style with type hints
- **Operators**: Always put spaces around operators: `x = 1 + 2` not `x=1+2`
- **Tests**: Use pytest style (no test classes), use fixtures and `pytest.mark.parametrize`

## Development Guidelines

### General Principles
- **Find root cause** before implementing solutions - fix problems, not symptoms
- **No workarounds** without asking - always identify and fix root causes
- **Separate logic from CLI** - CLI code should only handle I/O, not business logic
- **Linear flow preferred** - limit functions within scripts that control dataflow
- **Ask before complexity** - question simple problems with complex fixes

### Automatic Actions (No Approval Needed)
- **Fix linting errors** (flake8, ruff, black, isort, mypy) automatically
- **Create `__init__.py`** files in new module directories automatically
- **Create module folders** for new modules automatically
- **Edit multiple files** in single step for the same TODO task

### Git Conventions
- Keep commit messages straightforward - only describe code changes, no extraneous details

### Alerts (macOS specific)
- **Beep once** when done: `echo -ne '\007'` (only once, not continuously)
- **AppleScript notification**: `osascript -e 'tell application "System Events" to display dialog "Done with task X"'` (send ONE alert only)

## Development Strategy (from README)

Start small and iterate:
1. Begin with easy SQL questions
2. Evaluate early - submit 1-3 questions to public leaderboard to validate format
3. Incrementally add complexity:
   - Easy Wikipedia questions
   - Easy PDF questions
   - Then medium difficulty
   - Then hard questions

## Key Resources

- [Anthropic Claude API docs](https://docs.claude.com/en/docs/get-started#python)
- [Claude JSON Mode](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/develop/build-server)
- [Claude Embeddings](https://docs.claude.com/en/docs/build-with-claude/embeddings)