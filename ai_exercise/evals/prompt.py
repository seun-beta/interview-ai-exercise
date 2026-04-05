CORRECTNESS_PROMPT = """\
You are evaluating a RAG system's answer for factual correctness.

Ground truth:
{ground_truth}

RAG system's answer:
{rag_answer}

Does the RAG answer match the ground truth facts? Extra correct details beyond the ground truth are fine — do not penalize for being more detailed. Only penalize if the answer contradicts or misses the core facts.

Score from 1 to 5:
1 = Contradicts the ground truth
2 = Mostly wrong with some correct elements
3 = Partially correct but misses key facts
4 = Mostly correct with minor gaps
5 = Fully correct

Respond in exactly this format:
Score: <number>
Reason: <one line explanation>"""

COMPLETENESS_PROMPT = """\
You are evaluating a RAG system's answer for completeness.

Ground truth:
{ground_truth}

RAG system's answer:
{rag_answer}

Does the RAG answer cover the key information from the ground truth? It does not need to be word-for-word — it just needs to convey the same key points.

Score from 1 to 5:
1 = Misses all key information
2 = Covers very little
3 = Covers some key points but misses others
4 = Covers most key points
5 = Covers all key points

Respond in exactly this format:
Score: <number>
Reason: <one line explanation>"""

FAITHFULNESS_PROMPT = """\
You are evaluating a RAG system's answer for faithfulness.

Ground truth:
{ground_truth}

RAG system's answer:
{rag_answer}

Does the RAG answer only state things supported by the ground truth? Penalize if the answer invents facts, makes unsupported claims, or hallucinates details not present in the ground truth.

Score from 1 to 5:
1 = Mostly hallucinated content
2 = Significant unsupported claims
3 = Some unsupported claims mixed with grounded content
4 = Mostly grounded with minor unsupported details
5 = Fully grounded in the ground truth

Respond in exactly this format:
Score: <number>
Reason: <one line explanation>"""
