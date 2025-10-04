"""
Base classes for all agents in the multi-agent QA system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SourceInfo:
    """Information about a data source."""

    source_name: str
    source_type: str  # "db", "pdf", "wikipedia"
    page_number: Optional[int] = None


@dataclass
class AgentResponse:
    """
    Standardized response format for all agents.

    Attributes
    ----------
    value : Any
        The extracted value (number, string, bool, list, etc.)
    unit : Optional[str]
        Unit of measurement (e.g., "TWh", "tCO2eq", "count")
    sources : list[SourceInfo]
        List of sources used to derive the answer
    confidence : float
        Confidence score between 0.0 and 1.0
    metadata : dict
        Additional metadata (SQL query, reasoning, etc.)
    """

    value: Any
    unit: Optional[str] = None
    sources: list[SourceInfo] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "value": self.value,
            "unit": self.unit,
            "sources": [
                {
                    "source_name": s.source_name,
                    "source_type": s.source_type,
                    "page_number": s.page_number,
                }
                for s in self.sources
            ],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    All agents must implement the process() method which takes a question
    and returns an AgentResponse.
    """

    def __init__(self, name: str):
        """
        Initialize the agent.

        Parameters
        ----------
        name : str
            Agent name for logging and debugging
        """
        self.name = name

    @abstractmethod
    async def process(self, question: str) -> AgentResponse:
        """
        Process a question and return a structured response.

        Parameters
        ----------
        question : str
            Natural language question

        Returns
        -------
        AgentResponse
            Structured response with value, sources, and metadata
        """
        pass
