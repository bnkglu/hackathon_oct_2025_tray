"""
Router Agent for classifying questions and routing to appropriate specialized agents.
"""

from typing import Literal
from dataclasses import dataclass
from anthropic import Anthropic
import os


RouteType = Literal["db", "wikipedia", "rag", "hybrid"]


@dataclass
class RoutingDecision:
    """
    Decision from the router agent.

    Attributes
    ----------
    route_type : RouteType
        Which agent(s) to use: "db", "wikipedia", "rag", or "hybrid"
    confidence : float
        Confidence score (0.0 to 1.0)
    reasoning : str
        Explanation for the routing decision
    """

    route_type: RouteType
    confidence: float
    reasoning: str


class RouterAgent:
    """
    Routes questions to the appropriate specialized agent(s).

    Uses Claude to classify questions based on their content and determine
    which data sources are needed.
    """

    def __init__(self, anthropic_client: Anthropic):
        """
        Initialize the router agent.

        Parameters
        ----------
        anthropic_client : Anthropic
            Anthropic API client for Claude
        """
        self.name = "RouterAgent"
        self.client = anthropic_client

    async def route(self, question: str) -> RoutingDecision:
        """
        Classify a question and determine routing.

        Parameters
        ----------
        question : str
            Natural language question

        Returns
        -------
        RoutingDecision
            Routing decision with type, confidence, and reasoning
        """
        prompt = f"""Classify this question to determine which data source(s) should be used to answer it.

Question: {question}

Available data sources:
- "db": SQL database with energy and CO2 data (owid_energy_data, owid_co2_data tables)
  * Use for: country-level statistics, emissions, energy consumption, years, temporal comparisons
  * Keywords: CO2, emissions, energy, electricity, consumption, countries, years (2015-2021)

- "wikipedia": Wikipedia encyclopedia
  * Use for: general knowledge, company information, definitions, historical facts
  * Keywords: what is, who is, history, location, headquarters

- "rag": PDF annual reports (Erste Group, GSK, Swisscom)
  * Use for: company-specific data from annual reports, ESG metrics, financial data
  * Keywords: Erste, GSK, Swisscom, annual report, page numbers, specific company metrics

- "hybrid": Multiple sources needed
  * Use for: questions requiring combination of database + PDFs, or cross-source calculations

Respond with ONLY a JSON object in this exact format:
{{
  "route_type": "db" | "wikipedia" | "rag" | "hybrid",
  "confidence": 0.95,
  "reasoning": "brief explanation of why this routing was chosen"
}}"""

        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        import json

        response_text = response.content[0].text.strip()

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        return RoutingDecision(
            route_type=result["route_type"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
        )
