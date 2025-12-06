import argparse
import os
from .rag import RAGPipeline

def main():
    # Prefer model from .env if set, else safe default
    default_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    parser = argparse.ArgumentParser(description="RAG Chatbot CLI (Groq backend)")
    parser.add_argument("--k", type=int, default=6, help="Top-k documents to retrieve")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Groq model id (e.g., llama-3.1-8b-instant, llama-3.3-70b-versatile)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (0–1)")
    args = parser.parse_args()

    rag = RAGPipeline()
    print(f"RAG index loaded. Using model: {args.model} | top-k={args.k} | temperature={args.temperature}")
    print("Type your questions. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        answer, retrieved = rag.ask(q, top_k=args.k, model=args.model, temperature=args.temperature)
        print("\nAssistant:", answer)
        print("\nSources:")
        for i, d in enumerate(retrieved, 1):
            print(f"  {i}. {d.source}#{d.chunk_id} (score={d.score:.3f})")

if __name__ == "__main__":
    main()