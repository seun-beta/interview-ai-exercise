import asyncio
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ai_exercise.constants import SETTINGS, openai_client
from ai_exercise.evals.prompt import (COMPLETENESS_PROMPT, CORRECTNESS_PROMPT,
                                      FAITHFULNESS_PROMPT)

BASE_URL = "http://localhost:80"
EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

RATE_LIMIT_DELAY = 4

# Pricing per 1M tokens
COST_PER_1M_INPUT = 2.50
COST_PER_1M_OUTPUT = 10.00


def load_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


async def ask_rag(client: httpx.AsyncClient, question: str) -> dict[str, Any]:
    """Returns {"message": str, "token_usage": dict | None}."""
    resp = await client.post(f"{BASE_URL}/chat", json={"query": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


async def score_dimension(prompt_template: str, ground_truth: str, rag_answer: str) -> tuple[int, str, int, int]:
    """Score a single dimension. Returns (score, reason, prompt_tokens, completion_tokens)."""
    prompt = prompt_template.format(ground_truth=ground_truth, rag_answer=rag_answer)
    response = await openai_client.chat.completions.create(
        model=SETTINGS.openai_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    result = (response.choices[0].message.content or "").strip()

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    score = 0
    reason = ""
    for line in result.split("\n"):
        score_match = re.match(r"Score:\s*(\d)", line, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
        if line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return score, reason, prompt_tokens, completion_tokens


async def judge(ground_truth: str, rag_answer: str) -> dict[str, Any]:
    """Run 3 separate LLM judges concurrently — one per dimension."""
    correctness_result, completeness_result, faithfulness_result = await asyncio.gather(
        score_dimension(CORRECTNESS_PROMPT, ground_truth, rag_answer),
        score_dimension(COMPLETENESS_PROMPT, ground_truth, rag_answer),
        score_dimension(FAITHFULNESS_PROMPT, ground_truth, rag_answer),
    )

    correctness_score, correctness_reason, correctness_pt, correctness_ct = correctness_result
    completeness_score, completeness_reason, completeness_pt, completeness_ct = completeness_result
    faithfulness_score, faithfulness_reason, faithfulness_pt, faithfulness_ct = faithfulness_result

    return {
        "correctness_score": correctness_score,
        "completeness_score": completeness_score,
        "faithfulness_score": faithfulness_score,
        "correctness_reason": correctness_reason,
        "completeness_reason": completeness_reason,
        "faithfulness_reason": faithfulness_reason,
        "judge_prompt_tokens": correctness_pt + completeness_pt + faithfulness_pt,
        "judge_completion_tokens": correctness_ct + completeness_ct + faithfulness_ct,
    }


def check_confidence(scores: dict[str, Any]) -> str:
    """Flag questions where judges disagree significantly."""
    correctness = scores["correctness_score"]
    completeness = scores["completeness_score"]
    faithfulness = scores["faithfulness_score"]
    flags = []

    if faithfulness < 3 and correctness >= 4:
        flags.append(f"possible hallucination (correctness={correctness} faithfulness={faithfulness})")

    pairs = [
        (correctness, completeness, "correctness", "completeness"),
        (correctness, faithfulness, "correctness", "faithfulness"),
        (completeness, faithfulness, "completeness", "faithfulness"),
    ]
    for score_a, score_b, name_a, name_b in pairs:
        if abs(score_a - score_b) >= 3:
            flags.append(f"score disagreement ({name_a}={score_a} {name_b}={score_b})")

    return "; ".join(flags)


def save_results(results: list[dict[str, Any]], totals: dict[str, int]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"eval_{timestamp}.csv"

    fieldnames = [
        "id", "question", "category", "rag_answer",
        "correctness_score", "correctness_reason",
        "completeness_score", "completeness_reason",
        "faithfulness_score", "faithfulness_reason",
        "flagged",
        "judge_prompt_tokens", "judge_completion_tokens",
        "rag_prompt_tokens", "rag_completion_tokens",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        writer.writerow({
            "id": "TOTAL", "question": "", "category": "", "rag_answer": "",
            "correctness_score": "", "correctness_reason": "",
            "completeness_score": "", "completeness_reason": "",
            "faithfulness_score": "", "faithfulness_reason": "",
            "flagged": "",
            "judge_prompt_tokens": totals["judge_prompt"],
            "judge_completion_tokens": totals["judge_completion"],
            "rag_prompt_tokens": totals["rag_prompt"],
            "rag_completion_tokens": totals["rag_completion"],
        })

    return filepath


async def run() -> None:
    dataset = load_dataset()

    print(f"\nRunning eval on {len(dataset)} questions...\n")

    results = []
    total_judge_prompt = 0
    total_judge_completion = 0
    total_rag_prompt = 0
    total_rag_completion = 0

    async with httpx.AsyncClient() as client:
        for case in dataset:
            question = case["question"]
            ground_truth = case["ground_truth"]
            category = case["category"]

            print(f"Q{case['id']}: {question}")

            try:
                rag_response = await ask_rag(client, question)
                rag_answer = rag_response["message"]
                print(f"  RAG: {rag_answer[:150]}...")

                rag_prompt_tokens = 0
                rag_completion_tokens = 0
                token_usage = rag_response.get("token_usage")
                if token_usage:
                    for step in token_usage.values():
                        rag_prompt_tokens += step["prompt"]
                        rag_completion_tokens += step["completion"]
                    print(f"  RAG tokens: {rag_prompt_tokens} prompt + {rag_completion_tokens} completion")

                scores = await judge(ground_truth, rag_answer)
                flag = check_confidence(scores)

                total_judge_prompt += scores["judge_prompt_tokens"]
                total_judge_completion += scores["judge_completion_tokens"]
                total_rag_prompt += rag_prompt_tokens
                total_rag_completion += rag_completion_tokens

                print(f"  Correctness:  {scores['correctness_score']}/5 — {scores['correctness_reason']}")
                print(f"  Completeness: {scores['completeness_score']}/5 — {scores['completeness_reason']}")
                print(f"  Faithfulness: {scores['faithfulness_score']}/5 — {scores['faithfulness_reason']}")
                if flag:
                    print(f"  WARNING: {flag}")

            except Exception as e:
                print(f"  ERROR: {e}")
                rag_answer = ""
                scores = {
                    "correctness_score": 0, "completeness_score": 0, "faithfulness_score": 0,
                    "correctness_reason": f"ERROR: {e}",
                    "completeness_reason": f"ERROR: {e}",
                    "faithfulness_reason": f"ERROR: {e}",
                    "judge_prompt_tokens": 0, "judge_completion_tokens": 0,
                }
                flag = f"ERROR: {e}"
                rag_prompt_tokens = 0
                rag_completion_tokens = 0

            print()

            results.append({
                "id": case["id"],
                "question": question,
                "category": category,
                "rag_answer": rag_answer,
                "correctness_score": scores["correctness_score"],
                "correctness_reason": scores["correctness_reason"],
                "completeness_score": scores["completeness_score"],
                "completeness_reason": scores["completeness_reason"],
                "faithfulness_score": scores["faithfulness_score"],
                "faithfulness_reason": scores["faithfulness_reason"],
                "flagged": flag,
                "judge_prompt_tokens": scores["judge_prompt_tokens"],
                "judge_completion_tokens": scores["judge_completion_tokens"],
                "rag_prompt_tokens": rag_prompt_tokens,
                "rag_completion_tokens": rag_completion_tokens,
            })

            await asyncio.sleep(RATE_LIMIT_DELAY)

    # Save CSV
    totals = {
        "judge_prompt": total_judge_prompt,
        "judge_completion": total_judge_completion,
        "rag_prompt": total_rag_prompt,
        "rag_completion": total_rag_completion,
    }
    csv_path = save_results(results, totals)
    print(f"Results saved to {csv_path}")

    # Results table
    print()
    print("=" * 110)
    print(f"{'#':<3} {'Question':<45} {'Cat':<12} {'Cor':>4} {'Com':>4} {'Fai':>4}  Flag")
    print("-" * 110)
    for result in results:
        question_short = result["question"][:42] + "..." if len(result["question"]) > 45 else result["question"]
        print(f"{result['id']:<3} {question_short:<45} {result['category']:<12} {result['correctness_score']:>4} {result['completeness_score']:>4} {result['faithfulness_score']:>4}  {result['flagged'][:35]}")
    print("=" * 110)

    # Averages
    num_questions = len(results)
    avg_correctness = sum(r["correctness_score"] for r in results) / num_questions
    avg_completeness = sum(r["completeness_score"] for r in results) / num_questions
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / num_questions

    print(f"\nAverages:  Correctness={avg_correctness:.1f}/5  Completeness={avg_completeness:.1f}/5  Faithfulness={avg_faithfulness:.1f}/5")

    # Category breakdown
    categories: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        categories.setdefault(result["category"], []).append(result)

    print("\nBy category:")
    for category_name, category_results in categories.items():
        category_count = len(category_results)
        category_correctness = sum(r["correctness_score"] for r in category_results) / category_count
        category_completeness = sum(r["completeness_score"] for r in category_results) / category_count
        category_faithfulness = sum(r["faithfulness_score"] for r in category_results) / category_count
        print(f"  {category_name:<15} ({category_count} Qs)  Cor={category_correctness:.1f}  Com={category_completeness:.1f}  Fai={category_faithfulness:.1f}")

    # Cost tracking
    judge_total_tokens = total_judge_prompt + total_judge_completion
    judge_cost = (total_judge_prompt / 1_000_000 * COST_PER_1M_INPUT) + (total_judge_completion / 1_000_000 * COST_PER_1M_OUTPUT)
    print(f"\nJudge tokens: {total_judge_prompt:,} prompt + {total_judge_completion:,} completion = {judge_total_tokens:,} total (${judge_cost:.4f})")

    if total_rag_prompt > 0:
        rag_total_tokens = total_rag_prompt + total_rag_completion
        rag_cost = (total_rag_prompt / 1_000_000 * COST_PER_1M_INPUT) + (total_rag_completion / 1_000_000 * COST_PER_1M_OUTPUT)
        print(f"RAG tokens:   {total_rag_prompt:,} prompt + {total_rag_completion:,} completion = {rag_total_tokens:,} total (${rag_cost:.4f})")
        print(f"Total cost:   ${judge_cost + rag_cost:.4f}")
    else:
        print("RAG tokens:   not available (set DEBUG_MODE=true in .env to enable)")

    # Flagged questions
    flagged_results = [r for r in results if r["flagged"]]
    if flagged_results:
        print(f"\nFlagged questions ({len(flagged_results)}):")
        for result in flagged_results:
            print(f"  Q{result['id']}: {result['flagged']}")

    overall_score = (avg_correctness + avg_completeness + avg_faithfulness) / 3
    print(f"\nOverall score: {overall_score:.1f}/5 ({overall_score * 20:.0f}%)\n")


if __name__ == "__main__":
    asyncio.run(run())
