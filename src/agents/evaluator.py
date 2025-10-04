"""
Evaluator for final validation, formatting, and quality control.
"""

from typing import Any, Optional, Literal
import logging


AnswerType = Literal["float", "int", "string", "bool", "list"]
DifficultyLevel = Literal["easy", "medium", "hard", "extreme"]


class Evaluator:
    """
    Final quality control and formatting before submission.

    Responsibilities:
    - Validate answer types
    - Apply rounding rules
    - Detect hallucinations
    - Assign difficulty levels
    - Format to submission schema
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.name = "Evaluator"

    def _detect_answer_type(self, value: Any) -> AnswerType:
        """
        Detect the type of the answer value.

        Parameters
        ----------
        value : Any
            The answer value

        Returns
        -------
        AnswerType
            One of: "float", "int", "string", "bool", "list"
        """
        if value is None:
            return "float"  # Default for null answers
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, list):
            return "list"
        else:
            return "string"

    def _round_numeric(self, value: Any, answer_type: AnswerType) -> Any:
        """
        Apply rounding rules to numeric values.

        Parameters
        ----------
        value : Any
            The value to round
        answer_type : AnswerType
            The detected answer type

        Returns
        -------
        Any
            Rounded value (3 decimal places for floats)
        """
        if answer_type == "float" and value is not None:
            return round(float(value), 3)
        elif answer_type == "int" and value is not None:
            return int(round(float(value)))
        return value

    def _detect_hallucination(self, question: str, answer: Any, comment: str) -> bool:
        """
        Detect potential hallucinations in the answer.

        Parameters
        ----------
        question : str
            The original question
        answer : Any
            The proposed answer
        comment : str
            Reasoning/comment about the answer

        Returns
        -------
        bool
            True if hallucination detected, False otherwise
        """
        # Check for non-existent concepts
        hallucination_keywords = [
            "scope 5",
            "scope 4",
            "tier 4",
            "tier 5",
        ]

        question_lower = question.lower()
        for keyword in hallucination_keywords:
            if keyword in question_lower:
                logging.warning(f"Hallucination detected: {keyword} in question")
                return True

        return False

    def _classify_difficulty(
        self, question: str, sources: list[dict], comment: str
    ) -> DifficultyLevel:
        """
        Classify question difficulty.

        Parameters
        ----------
        question : str
            The question text
        sources : list[dict]
            List of sources used
        comment : str
            Reasoning/comment

        Returns
        -------
        DifficultyLevel
            One of: "easy", "medium", "hard", "extreme"
        """
        # Simple heuristics for difficulty classification
        num_sources = len(sources)
        question_lower = question.lower()

        # Multiple sources = harder
        if num_sources > 2:
            return "hard"

        # Cross-source calculations
        if "difference" in comment.lower() or "ratio" in comment.lower():
            return "medium"

        # Single source, direct lookup
        if num_sources == 1 and not any(
            op in question_lower for op in ["calculate", "compute", "ratio", "percentage"]
        ):
            return "easy"

        # Default
        return "medium"

    def evaluate(
        self,
        question: str,
        aggregated_result: dict,
        difficulty_override: Optional[DifficultyLevel] = None,
    ) -> dict:
        """
        Perform final evaluation and formatting.

        Parameters
        ----------
        question : str
            Original question
        aggregated_result : dict
            Result from Aggregator with value, unit, sources, comment
        difficulty_override : Optional[DifficultyLevel]
            Override automatic difficulty classification

        Returns
        -------
        dict
            Final formatted answer ready for submission
        """
        value = aggregated_result.get("value")
        unit = aggregated_result.get("unit")
        sources = aggregated_result.get("sources", [])
        comment = aggregated_result.get("comment", "")

        # Detect hallucination
        if self._detect_hallucination(question, value, comment):
            return {
                "question": question,
                "answer": None,
                "answer_type": "float",
                "unit": unit,
                "difficulty": "medium",
                "comment": "Hallucination detected: concept does not exist",
                "sources": None,
            }

        # Determine answer type
        answer_type = self._detect_answer_type(value)

        # Apply rounding
        rounded_value = self._round_numeric(value, answer_type)

        # Classify difficulty
        if difficulty_override:
            difficulty = difficulty_override
        else:
            difficulty = self._classify_difficulty(question, sources, comment)

        # Format final response
        return {
            "question": question,
            "answer": rounded_value,
            "answer_type": answer_type,
            "unit": unit,
            "difficulty": difficulty,
            "comment": comment,
            "sources": sources if sources else None,
        }
