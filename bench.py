#!/usr/bin/env python3
"""
Normierter Performance + Accuracy Benchmark fuer LLM endpoints.
Tests: throughput (tok/s), latency (TTFT), math accuracy.

WICHTIG: Dieses Script erzeugt deterministische Testdaten (random.seed(42)).
         Prompts und Mathe-Aufgaben sind fuer ALLE Konfigurationen identisch.
         Nicht aendern, damit Ergebnisse vergleichbar bleiben!

Usage:
  python3 bench.py --url http://localhost:8011 --model qwen3-coder-30b-bf16 --label "SGLang BF16 Vanilla (DGX)"
  python3 bench.py --url http://10.249.0.99:8011 --model qwen3-coder-30b-fp8 --label "vLLM FP8 (Spiegel 2)"

Prompts (fix):
  short:  "Was ist 7*8? Antworte nur mit der Zahl."                          max_tokens=20
  medium: "Erklaere in 3 Saetzen was ein Transformer ist."                   max_tokens=150
  long:   "Schreibe eine Python-Funktion die prueft ob eine Zahl prim ist."  max_tokens=400

Math (deterministic, seed=42):
  50 zufaellige Aufgaben (+, -, *) mit Zahlen 10-999, temperature=0

Context scaling (optional, --context):
  Decode throughput at 0/512/2K/8K/16K input context tokens
  Vergleichbar mit llama-benchy --pp/--tg Messungen
"""
import argparse
import json
import random
import sys
import time
import urllib.request

def chat(url, model, prompt, max_tokens=200, temperature=0):
    """Send chat request, return (content, completion_tokens, elapsed_s, ttft_s)."""
    data = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    content = body["choices"][0]["message"]["content"]
    tokens = body["usage"]["completion_tokens"]
    return content, tokens, elapsed

def perf_test(url, model, n=5):
    """Run throughput tests with varying output lengths."""
    prompts = [
        ("short", "Was ist 7*8? Antworte nur mit der Zahl.", 20),
        ("medium", "Erkläre in 3 Sätzen was ein Transformer ist.", 150),
        ("long", "Schreibe eine Python-Funktion die prüft ob eine Zahl prim ist. Erkläre den Code.", 400),
    ]
    results = []
    for label, prompt, max_tok in prompts:
        times = []
        tokens_list = []
        for _ in range(n):
            _, tokens, elapsed = chat(url, model, prompt, max_tok)
            tok_s = tokens / elapsed if elapsed > 0 else 0
            times.append(elapsed)
            tokens_list.append(tokens)
        avg_tok = sum(tokens_list) / len(tokens_list)
        avg_time = sum(times) / len(times)
        avg_toks = avg_tok / avg_time if avg_time > 0 else 0
        results.append({
            "type": label,
            "avg_tokens": avg_tok,
            "avg_time_s": round(avg_time, 2),
            "avg_tok_s": round(avg_toks, 1),
        })
        print(f"  {label:8s}: {avg_toks:6.1f} tok/s  ({avg_tok:.0f} tok in {avg_time:.2f}s, n={n})")
    return results

def context_test(url, model, contexts=None, tg=128, n=2):
    """Run decode throughput tests at varying context lengths (like llama-benchy)."""
    if contexts is None:
        contexts = [0, 512, 2048, 8192, 16384]

    # Build filler text: deterministic lorem-style padding
    random.seed(42)
    words = ("the quick brown fox jumps over a lazy dog near the river bank while "
             "clouds drift slowly across the wide blue sky and birds sing softly in "
             "the tall green trees beside the old stone wall that runs along ").split()

    results = []
    for ctx in contexts:
        # Build context prompt: filler text of ~ctx tokens (rough: 1 word ≈ 1.3 tokens)
        if ctx <= 0:
            filler = ""
            prompt = f"Continue writing a story about a fox. Write exactly {tg} words."
        else:
            n_words = int(ctx / 1.3)
            filler_words = []
            for i in range(n_words):
                filler_words.append(words[i % len(words)])
            filler = " ".join(filler_words)
            prompt = (f"Here is some context:\n\n{filler}\n\n"
                      f"Now continue writing this story. Write exactly {tg} words.")

        times = []
        tokens_list = []
        for run in range(n):
            _, tokens, elapsed = chat(url, model, prompt, max_tokens=tg + 50, temperature=0)
            times.append(elapsed)
            tokens_list.append(tokens)

        avg_tok = sum(tokens_list) / len(tokens_list)
        avg_time = sum(times) / len(times)
        avg_toks = avg_tok / avg_time if avg_time > 0 else 0
        results.append({
            "context": ctx,
            "avg_tokens": avg_tok,
            "avg_time_s": round(avg_time, 2),
            "avg_tok_s": round(avg_toks, 1),
        })
        print(f"  ctx={ctx:>6d}: {avg_toks:6.1f} tok/s  ({avg_tok:.0f} tok in {avg_time:.2f}s, n={n})")
    return results


def math_test(url, model, n=50):
    """Run math accuracy tests."""
    random.seed(42)
    correct = 0
    total = n
    errors = []

    for i in range(n):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op = random.choice(["+", "-", "*"])
        expected = eval(f"{a}{op}{b}")
        prompt = f"Berechne: {a} {op} {b} = ? Antworte NUR mit der Zahl, nichts anderes."

        content, _, _ = chat(url, model, prompt, max_tokens=30, temperature=0)
        # Extract number from response
        nums = []
        for word in content.replace(",", "").replace(".", "").split():
            try:
                nums.append(int(word))
            except ValueError:
                pass
        # Also try the whole content stripped
        try:
            nums.append(int(content.strip()))
        except ValueError:
            pass

        got_it = expected in nums
        if got_it:
            correct += 1
        else:
            errors.append(f"  {a}{op}{b}={expected}, got: {content.strip()[:40]}")

    pct = correct / total * 100
    print(f"  Math: {correct}/{total} ({pct:.0f}%)")
    if errors:
        for e in errors[:5]:
            print(e)
        if len(errors) > 5:
            print(f"  ... und {len(errors)-5} weitere Fehler")
    return {"correct": correct, "total": total, "pct": round(pct, 1)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--perf-rounds", type=int, default=5)
    parser.add_argument("--math-count", type=int, default=50)
    parser.add_argument("--skip-math", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--context", action="store_true", help="Run context-scaling test (decode at 0/512/2K/8K/16K)")
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[0, 512, 2048, 8192, 16384])
    parser.add_argument("--context-tg", type=int, default=128, help="Tokens to generate per context test")
    parser.add_argument("--context-rounds", type=int, default=2)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Benchmark: {args.label}")
    print(f"URL: {args.url}  Model: {args.model}")
    print(f"{'='*60}")

    results = {"label": args.label, "url": args.url, "model": args.model}

    if not args.skip_perf:
        print(f"\n--- Performance (n={args.perf_rounds}) ---")
        results["perf"] = perf_test(args.url, args.model, args.perf_rounds)

    if not args.skip_math:
        print(f"\n--- Math Accuracy (n={args.math_count}) ---")
        results["math"] = math_test(args.url, args.model, args.math_count)

    if args.context:
        print(f"\n--- Context Scaling (tg={args.context_tg}, n={args.context_rounds}) ---")
        results["context"] = context_test(
            args.url, args.model, args.context_sizes,
            args.context_tg, args.context_rounds)

    print(f"\n{'='*60}")
    # Output JSON for programmatic use
    print(f"\nJSON: {json.dumps(results)}")

if __name__ == "__main__":
    main()
