"""Demo satu file untuk absa-library."""
from __future__ import annotations

import sys

from absa_library import AbsaError, AspectAnalyzer

ENGINES = {
    "1": ("rule-based", "Rule-based, tanpa training"),
    "2": ("naive-bayes", "Naive Bayes, model bawaan"),
    "3": ("svm", "SVM, model bawaan"),
}


def print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"  aspek={row['aspect']:<14} "
            f"sentimen={row['sentiment']:<8} "
            f"confidence={row['confidence_score']:.4f}"
        )


def analyze(text: str, engine: str) -> None:
    analyzer = AspectAnalyzer(engine=engine)
    rows = analyzer.analyze(text)
    print(f"\nTeks   : {text}")
    print(f"Engine : {engine}")
    if not rows:
        print("  tidak ada hasil")
        return
    print_rows(rows)


def demo() -> None:
    samples = [
        "Barangnya bagus banget, tapi kurirnya lama",
        "Aplikasinya sering error, tp harganya murah",
        "Makanannya enak tapi pelayanannya lambat",
        "Kameranya jernih, baterainya cepat habis",
    ]
    for engine in ("rule-based", "svm"):
        print("=" * 68)
        print(f"ENGINE: {engine}")
        for text in samples:
            analyze(text, engine)


def choose_engine() -> str:
    print("\nPilih engine:")
    for key, (_, description) in ENGINES.items():
        print(f"  {key}. {description}")
    choice = input("Pilihan [1]: ").strip() or "1"
    if choice not in ENGINES:
        print("Pilihan tidak valid, memakai rule-based.")
        choice = "1"
    return ENGINES[choice][0]


def interactive() -> None:
    print("Demo ABSA. Ketik q lalu Enter untuk keluar.")
    engine = choose_engine()
    print(f"Engine aktif: {engine}")
    while True:
        text = input("\nTeks: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue
        try:
            analyze(text, engine)
        except AbsaError as exc:
            print(f"Error: {exc}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
        return
    if len(sys.argv) > 1:
        analyze(" ".join(sys.argv[1:]), "rule-based")
        return
    interactive()


if __name__ == "__main__":
    main()