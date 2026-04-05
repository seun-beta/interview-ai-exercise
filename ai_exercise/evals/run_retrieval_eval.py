import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from chromadb.api.types import Embeddable, EmbeddingFunction

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.retrieval.retrieval import get_relevant_chunks, rewrite_query
from ai_exercise.retrieval.vector_store import create_collection

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

RATE_LIMIT_DELAY = 4


def load_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


async def run() -> None:
    dataset = load_dataset()
    collection = create_collection(
        chroma_client, cast(EmbeddingFunction[Embeddable], openai_ef), SETTINGS.collection_name
    )

    doc_count = collection.count()
    print(f"\nCollection has {doc_count} documents")
    print(f"Running retrieval eval on {len(dataset)} questions (k={SETTINGS.k_neighbors})...\n")

    results = []
    hits = 0
    evaluated = 0

    for case in dataset:
        question = case["question"]
        expected_path = case.get("expected_path")

        print(f"Q{case['id']}: {question}")

        rewritten, _ = await rewrite_query(
            client=openai_client, query=question, model=SETTINGS.openai_model
        )
        print(f"  Rewritten: {rewritten[:100]}")

        chunks = await get_relevant_chunks(
            collection=collection, query=rewritten, k=SETTINGS.k_neighbors
        )

        if expected_path:
            hit = any(expected_path in chunk for chunk in chunks)
            evaluated += 1
            if hit:
                hits += 1
            print(f"  Expected path: {expected_path} -> {'HIT' if hit else 'MISS'}")
        else:
            hit = None
            print(f"  Expected path: N/A (skipped)")

        for i, chunk in enumerate(chunks):
            first_line = chunk.split("\n")[0]
            print(f"  Chunk {i+1}: {first_line[:80]}")

        print()

        results.append({
            "id": case["id"],
            "question": question,
            "category": case["category"],
            "expected_path": expected_path or "",
            "retrieval_hit": "" if hit is None else ("HIT" if hit else "MISS"),
            "rewritten_query": rewritten,
            "chunk_1": chunks[0].split("\n")[0] if len(chunks) > 0 else "",
            "chunk_2": chunks[1].split("\n")[0] if len(chunks) > 1 else "",
            "chunk_3": chunks[2].split("\n")[0] if len(chunks) > 2 else "",
            "chunk_4": chunks[3].split("\n")[0] if len(chunks) > 3 else "",
            "chunk_5": chunks[4].split("\n")[0] if len(chunks) > 4 else "",
        })

        await asyncio.sleep(RATE_LIMIT_DELAY)

    # Save CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"retrieval_eval_{timestamp}.csv"

    fieldnames = [
        "id", "question", "category", "expected_path", "retrieval_hit",
        "rewritten_query", "chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_path}")

    # Summary table
    print()
    print("=" * 80)
    print(f"{'#':<3} {'Question':<45} {'Expected':<25} {'Result'}")
    print("-" * 80)
    for r in results:
        q = r["question"][:42] + "..." if len(r["question"]) > 45 else r["question"]
        print(f"{r['id']:<3} {q:<45} {r['expected_path']:<25} {r['retrieval_hit']}")
    print("=" * 80)

    if evaluated > 0:
        print(f"\nRetrieval accuracy: {hits}/{evaluated} ({100 * hits // evaluated}%)")
    print(f"Questions evaluated: {evaluated} (skipped {len(dataset) - evaluated} with no expected_path)\n")


if __name__ == "__main__":
    asyncio.run(run())
