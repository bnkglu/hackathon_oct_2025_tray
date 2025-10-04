"""
DB Agent for answering questions using SQL database queries.
"""

from typing import Optional
from src.agents.base import BaseAgent, AgentResponse, SourceInfo
from src.util.client import MCPClient
from anthropic import Anthropic
import json
import logging


class DBAgent(BaseAgent):
    """
    Agent for querying SQL databases to answer questions.

    Uses the existing database MCP server (src/mcp_servers/database.py) which provides:
    - Tool: query_database(sql_query) -> list[list[Any]]
    - Resource: schema://{table_name} -> str (table schema)
    - Resource: tables:// -> list[str] (list of tables)

    Process:
    1. Get available tables and schemas from MCP resources
    2. Use Claude to generate SQL query based on question + schemas
    3. Execute query via query_database tool
    4. Parse and return result
    """

    def __init__(self, database_client: MCPClient, anthropic_client: Anthropic):
        """
        Initialize the DB Agent.

        Parameters
        ----------
        database_client : MCPClient
            MCP client connected to the database server (src.mcp_servers.database)
        anthropic_client : Anthropic
            Anthropic API client for Claude
        """
        super().__init__("DBAgent")
        self.db_client = database_client
        self.claude = anthropic_client
        self.schema_cache = {}

    async def _get_table_schemas(self) -> dict[str, str]:
        """
        Get schemas for all available tables using MCP resources.

        Returns
        -------
        dict[str, str]
            Dictionary mapping table names to their schema descriptions
        """
        if self.schema_cache:
            return self.schema_cache

        try:
            # Get list of tables using the tables:// resource
            resources = await self.db_client.list_resources()
            # FIX: Convert AnyUrl to string for comparison
            tables_resource = [r for r in resources if "tables://" in str(r.uri)]

            if tables_resource:
                tables_result = await self.db_client.session.read_resource(
                    tables_resource[0].uri
                )
                # Parse table names from the resource content
                tables_text = tables_result.contents[0].text
                # The resource returns a list, parse it
                table_names = eval(tables_text) if tables_text else []
                logging.info(f"Found {len(table_names)} tables: {table_names}")
            else:
                # Fallback to known tables
                logging.warning("Could not find tables:// resource, using fallback")
                table_names = ["owid_energy_data", "owid_co2_data"]

            # Get schema for each table using schema://{table_name} resource
            schemas = {}
            for table_name in table_names:
                if isinstance(table_name, str) and not table_name.startswith("Error"):
                    try:
                        schema_result = await self.db_client.session.read_resource(
                            f"schema://{table_name}"
                        )
                        schema_text = schema_result.contents[0].text
                        schemas[table_name] = schema_text
                        logging.info(f"Retrieved schema for {table_name}: {len(schema_text)} chars")
                    except Exception as e:
                        logging.error(f"Could not get schema for {table_name}: {e}")
                        schemas[table_name] = f"Table: {table_name} (schema unavailable)"

            self.schema_cache = schemas
            return schemas

        except Exception as e:
            logging.error(f"Error getting table schemas: {e}")
            # Return minimal fallback schemas
            return {
                "owid_energy_data": "Table: owid_energy_data (columns include country, year, fossil_fuel_consumption, low_carbon_electricity, etc.)",
                "owid_co2_data": "Table: owid_co2_data (columns include country, year, co2, consumption_co2, etc.)",
            }

    async def _generate_sql(self, question: str, schemas: dict[str, str]) -> str:
        """
        Generate SQL query using Claude.

        Parameters
        ----------
        question : str
            Natural language question
        schemas : dict[str, str]
            Available table schemas

        Returns
        -------
        str
            SQL query
        """
        schema_text = "\n\n".join([f"{name}:\n{schema}" for name, schema in schemas.items()])

        prompt = f"""You are a SQL expert for SQL Server. Generate a SQL query to answer this question.

Available database schemas:
{schema_text}

Question: {question}

Important guidelines:
- This is SQL Server, not PostgreSQL or MySQL
- Use proper SQL Server syntax (e.g., TOP instead of LIMIT)
- Use table and column names exactly as shown in schemas (case-sensitive)
- For temporal comparisons, calculate the SIGNED difference (a - b), NOT absolute value
- DO NOT use ABS() unless the question explicitly asks for "absolute value"
- "difference" or "absolute difference" means SIGNED difference, which can be negative
- For country='World', data is already aggregated - use direct SELECT, NOT SUM()
- For multi-country queries (excluding 'World'), use WHERE IN or aggregate functions
- Use subqueries or CTEs for complex calculations
- Return clean numeric results when possible

Common query patterns:
- Difference between years (SIGNED, can be negative):
  SELECT (SELECT value FROM table WHERE year=2020) - (SELECT value FROM table WHERE year=2015) AS difference

- World queries (NO SUM needed):
  SELECT column FROM table WHERE country='World' AND year=2021

- Multi-country aggregation (SUM multiple countries):
  SELECT SUM(column) FROM table WHERE country IN ('France', 'Germany') AND year=2021

- Column arithmetic:
  SELECT (col1 - col2) AS difference FROM table WHERE conditions

CRITICAL: Never use ABS() for "difference" or "absolute difference" - these should return signed values.
Only use ABS() if question explicitly says "absolute value of X".

Respond with ONLY the SQL query, no explanation or markdown."""

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        sql_query = response.content[0].text.strip()

        # Clean up markdown formatting if present
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        return sql_query

    async def _execute_query(self, sql_query: str):
        """
        Execute SQL query via MCP tool.

        Parameters
        ----------
        sql_query : str
            SQL query to execute

        Returns
        -------
        Any
            Query results - could be list, float, int, or string
        """
        result = await self.db_client.call_tool("query_database", {"sql_query": sql_query})

        # The MCP tool returns results in result.content[0].text
        if result.content:
            result_text = result.content[0].text
            # The result could be JSON, a raw number, or a string
            # Try to parse as various types
            try:
                # First try JSON
                result_data = json.loads(result_text)
                return result_data
            except (json.JSONDecodeError, ValueError):
                # Try as a number
                try:
                    return float(result_text)
                except ValueError:
                    # Return as-is (string)
                    return result_text
        return None

    def _parse_result(
        self, query_result, question: str, sql_query: str
    ) -> tuple[Optional[float], Optional[str], str]:
        """
        Parse query result to extract value and unit.

        Parameters
        ----------
        query_result : Any
            Raw query result from database (could be list, float, int, or string)
        question : str
            Original question for context
        sql_query : str
            SQL query that was executed

        Returns
        -------
        tuple[Optional[float], Optional[str], str]
            (value, unit, table_name)
        """
        # Handle different result formats
        value = None

        if query_result is None:
            logging.warning("Empty query result")
            return None, None, "unknown"

        # If it's already a number, use it directly
        if isinstance(query_result, (int, float)):
            value = float(query_result)

        # If it's a list of lists (standard format)
        elif isinstance(query_result, list):
            if len(query_result) == 0:
                logging.warning("Empty query result list")
                return None, None, "unknown"

            # Check for error
            if isinstance(query_result[0], list) and query_result[0][0] == "Error":
                logging.error(f"Database error: {query_result[0][1]}")
                return None, None, "unknown"

            # Extract value (first row, first column for aggregations)
            if isinstance(query_result[0], list):
                value = query_result[0][0]
            else:
                value = query_result[0]

        # If it's a string, try to convert
        elif isinstance(query_result, str):
            # Check for error string
            if query_result.lower() == "error" or query_result.startswith("Error"):
                logging.error(f"Database returned error for SQL: {sql_query}")
                return None, None, "unknown"

            try:
                value = float(query_result)
            except ValueError:
                logging.warning(f"Could not convert string result to float: {query_result}")
                return None, None, "unknown"

        # Try to convert to float if not already
        if value is not None and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                logging.warning(f"Could not convert result to float: {value}")

        # Infer unit from question and query
        unit = None
        question_lower = question.lower()
        sql_lower = sql_query.lower()

        # Infer units based on question keywords
        if "co2" in question_lower:
            if "million" in question_lower or "mt" in question_lower:
                unit = "million tonnes"
            else:
                unit = "tCO2eq"
        elif (
            "electricity" in question_lower
            or "energy" in question_lower
            or "consumption" in question_lower
        ):
            unit = "TWh"
        elif "count" in question_lower or "number" in question_lower:
            unit = "count"

        # Determine source table from SQL query
        table_name = "owid_co2_data"  # default
        if "energy" in sql_lower or "fossil_fuel_consumption" in sql_lower:
            table_name = "owid_energy_data"
        elif "co2" in sql_lower:
            table_name = "owid_co2_data"

        return value, unit, table_name

    async def process(self, question: str) -> AgentResponse:
        """
        Process a question using database queries.

        Parameters
        ----------
        question : str
            Natural language question

        Returns
        -------
        AgentResponse
            Structured response with value, sources, and metadata
        """
        try:
            # Step 1: Get table schemas
            schemas = await self._get_table_schemas()

            # Step 2: Generate SQL query using Claude
            sql_query = await self._generate_sql(question, schemas)
            logging.info(f"Generated SQL: {sql_query}")

            # Step 3: Execute query via MCP tool
            query_result = await self._execute_query(sql_query)
            logging.info(f"Query result: {query_result}")

            # Step 4: Parse result
            value, unit, table_name = self._parse_result(query_result, question, sql_query)

            # Create response
            return AgentResponse(
                value=value,
                unit=unit,
                sources=[SourceInfo(source_name=table_name, source_type="db")],
                confidence=0.9,
                metadata={
                    "sql_query": sql_query,
                    "raw_result": query_result,
                    "reasoning": f"Executed SQL query on {table_name}",
                },
            )

        except Exception as e:
            logging.error(f"Error in DBAgent.process: {e}", exc_info=True)
            return AgentResponse(
                value=None,
                unit=None,
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)},
            )
