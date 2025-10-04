"""
Evaluation script for testing DB Agent on database questions.

Loads questions from public_questions.json, filters to DB-only questions,
runs the agent, and compares results with expected answers.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from src.util.utils import get_root_dir
from src.agent import Agent

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def evaluate_db_questions(
    limit: Optional[int] = None, tolerance: float = 0.001, verbose: bool = True
):
    """
    Evaluate DB Agent performance on database questions.

    Parameters
    ----------
    limit : Optional[int]
        Limit number of questions to evaluate (None = all DB questions)
    tolerance : float
        Acceptable tolerance for numeric answers (default: 0.001)
    verbose : bool
        Print detailed results

    Returns
    -------
    dict
        Evaluation metrics and results
    """
    # Load questions
    questions_path = get_root_dir() / "data" / "public_questions.json"
    with open(questions_path, "r") as f:
        all_questions = json.load(f)

    # Filter to DB-only questions
    db_questions = {
        q_id: q_data
        for q_id, q_data in all_questions.items()
        if q_data.get("sources")
        and any(s.get("source_type") == "db" for s in q_data["sources"])
    }

    # Apply limit if specified
    if limit:
        db_question_ids = list(db_questions.keys())[:limit]
    else:
        db_question_ids = list(db_questions.keys())

    logging.info(f"Evaluating on {len(db_question_ids)} DB questions\n")

    # Initialize agent
    agent = Agent()
    results = []

    try:
        await agent.initialise_servers()

        # Process each question
        for i, q_id in enumerate(db_question_ids, 1):
            question_data = db_questions[q_id]
            question_text = question_data["question"]
            expected_answer = question_data.get("answer")
            expected_type = question_data.get("answer_type")

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Question {i}/{len(db_question_ids)} (ID: {q_id})")
                print(f"{'=' * 70}")
                print(f"Q: {question_text}")

            # Get answer from agent
            answer_dict = await agent.answer_question(question_text)
            predicted_answer = answer_dict.get("answer")

            # Check correctness
            is_correct = False
            error = None

            if expected_answer is None and predicted_answer is None:
                is_correct = True  # Both null (e.g., hallucination questions)
            elif expected_answer is not None and predicted_answer is not None:
                if expected_type in ["float", "int"]:
                    # Numeric comparison with tolerance
                    try:
                        diff = abs(float(predicted_answer) - float(expected_answer))
                        is_correct = diff <= tolerance
                        error = diff if not is_correct else None
                    except (ValueError, TypeError):
                        is_correct = False
                        error = "Type conversion error"
                else:
                    # Exact match for strings/bools
                    is_correct = predicted_answer == expected_answer
                    error = None if is_correct else "Mismatch"
            else:
                is_correct = False
                error = "One is None, other is not"

            # Store result
            result = {
                "question_id": q_id,
                "question": question_text,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "correct": is_correct,
                "error": error,
                "metadata": answer_dict.get("metadata", {}),
            }
            results.append(result)

            if verbose:
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                print(f"Expected: {expected_answer}")
                print(f"Predicted: {predicted_answer}")
                print(f"Status: {status}")
                if error:
                    print(f"Error: {error}")

    finally:
        # Cleanup
        if agent.database_client:
            await agent.database_client.cleanup()
        if agent.wikipedia_client:
            await agent.wikipedia_client.cleanup()

    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Print summary
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total questions: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save detailed results
    output_path = get_root_dir() / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {"summary": {"total": total, "correct": correct, "accuracy": accuracy}, "results": results},
            f,
            indent=2,
        )

    logging.info(f"\nDetailed results saved to {output_path}")

    return {"summary": {"total": total, "correct": correct, "accuracy": accuracy}, "results": results}


async def main():
    """Run evaluation on DB questions."""
    import sys

    # Parse command line arguments
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    # Run evaluation (default: first 10 questions for testing)
    await evaluate_db_questions(limit=limit or 10, verbose=True)


if __name__ == "__main__":
    asyncio.run(main())
