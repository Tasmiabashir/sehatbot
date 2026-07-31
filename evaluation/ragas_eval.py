"""
ragas_eval.py — SehatBot Evaluation Suite

Runs the real SehatBot pipeline over a test set and measures quality with:
  • ROUGE       : rouge1, rouge2, rougeL (overlap with the reference answer)
  • BLEU        : n-gram precision vs the reference answer
  • LLM-as-judge: an LLM rates each answer 1-5 for accuracy and helpfulness
  • RAGAS       : Faithfulness, Context Precision, Context Recall, Answer Relevancy

All answers are produced by SehatBot itself (real hybrid RAG + tools).

Run from the project root:
  python evaluation/ragas_eval.py          # 5-question set (token-safe)
  python evaluation/ragas_eval.py full     # full 10-question set
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

# Make the backend importable so we reuse the REAL pipeline
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "backend"))

from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

from rag import search                     # real hybrid search
import tools                               # real tool functions

TOOL_FOR_MODE = {
    "symptom_checker": tools.check_symptoms,
    "lab_report"     : tools.analyze_lab_report,
    "medicine_safety": tools.check_drug_interaction,
    "mental_health"  : tools.assess_mental_health,
    "diet_advisor"   : tools.get_diet_plan,
    "emergency"      : tools.get_emergency_guide,
}


def run_sehatbot(mode: str, question: str) -> Tuple[str, List[str]]:
    """Produce a real answer + the retrieved context for one question."""
    docs = search(mode, question)
    context = [d.page_content for d in docs]
    func = TOOL_FOR_MODE[mode]
    try:
        answer = func.invoke(question)     # @tool-decorated
    except Exception:
        answer = func(question)            # plain call fallback
    return str(answer), context


def compute_rouge(answers: List[str], ground_truths: List[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for ans, gt in zip(answers, ground_truths):
        s = scorer.score(gt, ans)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = max(len(answers), 1)
    return {"rouge1": round(r1 / n, 4), "rouge2": round(r2 / n, 4), "rougeL": round(rl / n, 4)}


def compute_bleu(answers: List[str], ground_truths: List[str]) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    total: float = 0.0
    for ans, gt in zip(answers, ground_truths):
        reference: List[List[str]] = [gt.lower().split()]
        candidate: List[str] = ans.lower().split()
        score: float = float(sentence_bleu(reference, candidate, smoothing_function=smooth))
        total += score
    n = max(len(answers), 1)
    return round(total / n, 4)


def compute_llm_judge(questions: List[str], answers: List[str], ground_truths: List[str]) -> float:
    from langchain_google_genai import ChatGoogleGenerativeAI
    judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    scores: List[int] = []
    for q, ans, gt in zip(questions, answers, ground_truths):
        jp = (
            "You are a medical answer evaluator. Rate the AI answer from 1 to 5. "
            "5 = accurate, safe, and helpful; 1 = wrong or unsafe.\n"
            f"Question: {q}\nReference answer: {gt}\nAI answer: {ans}\n"
            "Reply with ONLY a single number from 1 to 5."
        )
        try:
            raw_content = judge.invoke(jp).content
            raw: str = raw_content if isinstance(raw_content, str) else str(raw_content)
            digits = "".join(ch for ch in raw if ch.isdigit())
            scores.append(int(digits[0]) if digits else 0)
        except Exception as e:
            print("  judge error:", e)
            scores.append(0)
    n = max(len(scores), 1)
    return round(sum(scores) / n, 2)


def compute_ragas(questions: List[str], answers: List[str],
                  contexts: List[List[str]], ground_truths: List[str]) -> dict:
    """RAGAS with Groq judge + local embeddings. Returns an error note if the
    RAGAS library fails to import or run (known version issues)."""
    try:
        # ── SHIM: old ragas versions import a module path that newer
        # langchain-community deleted. Recreate that path and point it at the
        # installed langchain-google-vertexai package so the import succeeds.
        import types
        import importlib
        try:
            importlib.import_module("langchain_community.chat_models.vertexai")
        except ModuleNotFoundError:
            stub = types.ModuleType("langchain_community.chat_models.vertexai")
            try:
                from langchain_google_vertexai import ChatVertexAI as _CVA
            except Exception:
                class _CVA:                      # placeholder; never actually used
                    pass
            stub.ChatVertexAI = _CVA            # type: ignore[attr-defined]
            sys.modules["langchain_community.chat_models.vertexai"] = stub

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", max_retries=2))
        ragas_emb = LangchainEmbeddingsWrapper(
            SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        ds = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        metrics = [faithfulness, context_precision, context_recall, answer_relevancy]
        for m in metrics:
            if hasattr(m, "llm"):
                m.llm = ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = ragas_emb

        result = evaluate(dataset=ds, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb)

        out: dict = {}
        df = result.to_pandas()
        for col in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
            if col in df.columns:
                out[col] = round(float(df[col].mean()), 4)
        return out
    except Exception as e:
        print("  RAGAS error (metrics skipped):", e)
        return {"error": str(e)[:160]}


def main() -> None:
    test_file = "test_cases.json" if (len(sys.argv) > 1 and sys.argv[1] == "full") else "test_cases_small.json"
    cases = json.load(open(ROOT / "evaluation" / test_file, encoding="utf-8"))
    print(f"Running SehatBot over {len(cases)} test cases ({test_file})...\n")

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {c['mode']}: {c['question'][:50]}...")
        ans, ctx = run_sehatbot(c["mode"], c["question"])
        questions.append(c["question"])
        answers.append(ans)
        contexts.append(ctx if ctx else ["(no context retrieved)"])
        ground_truths.append(c["ground_truth"])

    print("\n-- Computing ROUGE --")
    rouge_results = compute_rouge(answers, ground_truths)

    print("-- Computing BLEU --")
    bleu_score = compute_bleu(answers, ground_truths)

    print("-- Running LLM-as-judge (Gemini) --")
    llm_judge_avg = compute_llm_judge(questions, answers, ground_truths)

    print("-- Running RAGAS (Groq judge + local embeddings) --")
    ragas_results = compute_ragas(questions, answers, contexts, ground_truths)

    report = {
        "test_cases": len(cases),
        "RAGAS": ragas_results,
        "ROUGE": rouge_results,
        "BLEU": bleu_score,
        "LLM_as_judge_avg_out_of_5": llm_judge_avg,
    }

    print("\n" + "=" * 50)
    print("SEHATBOT EVALUATION RESULTS")
    print("=" * 50)
    print(json.dumps(report, indent=2))

    out_path = ROOT / "evaluation" / "results.json"
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()