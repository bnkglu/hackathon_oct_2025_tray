"""
Quick test script to run questions 40, 41, 42 (DB questions)
"""
import asyncio
import json
from src.util.utils import get_root_dir
from src.agent import Agent


async def test_specific_questions():
    """Test questions 40, 41, 42 specifically."""
    # Load all questions
    questions_path = get_root_dir() / "data" / "public_questions.json"
    with open(questions_path, "r") as f:
        all_questions = json.load(f)

    # Get specific DB questions
    test_ids = ["40", "41", "42"]

    agent = Agent()
    results = {}

    try:
        await agent.initialise_servers()

        for q_id in test_ids:
            question_data = all_questions[q_id]
            question_text = question_data["question"]
            expected = question_data["answer"]

            print(f"\n{'=' * 70}")
            print(f"Question {q_id}")
            print(f"{'=' * 70}")
            print(f"Q: {question_text}")
            print(f"Expected: {expected} {question_data.get('unit', '')}")

            answer_dict = await agent.answer_question(question_text)

            print(f"Predicted: {answer_dict['answer']} {answer_dict.get('unit', '')}")
            print(f"Sources: {answer_dict.get('sources', [])}")

            # Check accuracy
            if answer_dict['answer'] is not None and expected is not None:
                diff = abs(float(answer_dict['answer']) - float(expected))
                status = "✅ CORRECT" if diff < 0.01 else "❌ INCORRECT"
                print(f"Difference: {diff:.6f}")
                print(f"Status: {status}")

            results[q_id] = answer_dict

    finally:
        if agent.database_client:
            await agent.database_client.cleanup()
        if agent.wikipedia_client:
            await agent.wikipedia_client.cleanup()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    correct = 0
    for q_id in test_ids:
        expected = all_questions[q_id]["answer"]
        predicted = results[q_id]["answer"]
        if predicted is not None and expected is not None:
            diff = abs(float(predicted) - float(expected))
            if diff < 0.01:
                correct += 1
                print(f"Q{q_id}: ✅ CORRECT")
            else:
                print(f"Q{q_id}: ❌ INCORRECT (diff: {diff:.6f})")

    print(f"\nAccuracy: {correct}/{len(test_ids)} ({correct/len(test_ids)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(test_specific_questions())
