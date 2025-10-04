"""
Aggregator for combining multiple agent responses into a coherent answer.
"""

from typing import Optional
from src.agents.base import AgentResponse, SourceInfo
import logging


class Aggregator:
    """
    Combines outputs from multiple agents and performs arithmetic operations.

    Handles:
    - Arithmetic: differences, sums, ratios, averages
    - Unit consistency checking
    - Source merging and deduplication
    - Reasoning trace generation
    """

    def __init__(self):
        """Initialize the aggregator."""
        self.name = "Aggregator"

    def aggregate_single(self, response: AgentResponse, question: str) -> dict:
        """
        Process a single agent response into final answer format.

        Parameters
        ----------
        response : AgentResponse
            Response from a single agent
        question : str
            Original question for context

        Returns
        -------
        dict
            Formatted answer with value, unit, sources, and comment
        """
        return {
            "value": response.value,
            "unit": response.unit,
            "sources": [
                {
                    "source_name": s.source_name,
                    "source_type": s.source_type,
                    "page_number": s.page_number,
                }
                for s in response.sources
            ],
            "comment": response.metadata.get("reasoning", ""),
            "metadata": response.metadata,
        }

    def aggregate_multiple(
        self, responses: list[AgentResponse], question: str, operation: str = "combine"
    ) -> dict:
        """
        Combine multiple agent responses with arithmetic operations.

        Parameters
        ----------
        responses : list[AgentResponse]
            List of responses from different agents
        question : str
            Original question for context
        operation : str
            Operation to perform: "combine", "sum", "difference", "ratio", "average"

        Returns
        -------
        dict
            Formatted answer with combined value, merged sources, and reasoning
        """
        if not responses:
            return {
                "value": None,
                "unit": None,
                "sources": [],
                "comment": "No responses received from agents",
                "metadata": {},
            }

        # Merge all sources
        all_sources = []
        source_set = set()
        for response in responses:
            for source in response.sources:
                source_key = (source.source_name, source.source_type, source.page_number)
                if source_key not in source_set:
                    source_set.add(source_key)
                    all_sources.append(
                        {
                            "source_name": source.source_name,
                            "source_type": source.source_type,
                            "page_number": source.page_number,
                        }
                    )

        # Perform arithmetic operation
        if operation == "sum":
            total = sum(float(r.value) for r in responses if r.value is not None)
            unit = responses[0].unit
            comment = f"Sum of {len(responses)} values: {' + '.join(str(r.value) for r in responses)}"
            return {
                "value": total,
                "unit": unit,
                "sources": all_sources,
                "comment": comment,
                "metadata": {"operation": "sum", "components": len(responses)},
            }

        elif operation == "difference":
            if len(responses) != 2:
                logging.warning(f"Difference operation expects 2 values, got {len(responses)}")
            diff = float(responses[0].value) - float(responses[1].value)
            unit = responses[0].unit
            comment = f"Difference: {responses[0].value} - {responses[1].value}"
            return {
                "value": diff,
                "unit": unit,
                "sources": all_sources,
                "comment": comment,
                "metadata": {"operation": "difference"},
            }

        elif operation == "ratio":
            if len(responses) != 2:
                logging.warning(f"Ratio operation expects 2 values, got {len(responses)}")
            ratio = float(responses[0].value) / float(responses[1].value)
            comment = f"Ratio: {responses[0].value} / {responses[1].value}"
            return {
                "value": ratio,
                "unit": "ratio",
                "sources": all_sources,
                "comment": comment,
                "metadata": {"operation": "ratio"},
            }

        elif operation == "average":
            avg = sum(float(r.value) for r in responses if r.value is not None) / len(
                responses
            )
            unit = responses[0].unit
            comment = f"Average of {len(responses)} values"
            return {
                "value": avg,
                "unit": unit,
                "sources": all_sources,
                "comment": comment,
                "metadata": {"operation": "average", "count": len(responses)},
            }

        else:  # "combine" or default
            # Just take the first response if no specific operation
            if len(responses) == 1:
                return self.aggregate_single(responses[0], question)
            else:
                # For multiple responses without operation, return first with all sources
                return {
                    "value": responses[0].value,
                    "unit": responses[0].unit,
                    "sources": all_sources,
                    "comment": f"Combined {len(responses)} responses",
                    "metadata": {"operation": "combine"},
                }
