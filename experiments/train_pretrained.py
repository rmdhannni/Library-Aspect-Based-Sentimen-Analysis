"""Train the bundled Naive Bayes and SVM models.

The models are trained on the main research dataset (Indonesian AI Chatbots
reviews, 3 classes) so that ``AspectAnalyzer(engine="svm")`` works out of the
box. Run from the repository root::

    python experiments/train_pretrained.py

Artifacts are written to ``src/absa_library/data/pretrained`` and the metric
report to ``experiments/output/pretrained_metrics.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from absa_library import NaiveBayesEngine, SVMEngine, TextPreprocessor
from absa_library.types import normalize_label

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "raw" / "ai_chatbots_reviews.csv"
OUT = ROOT / "src" / "absa_library" / "data" / "pretrained"
REPORT = ROOT / "experiments" / "pretrained_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 15000


def load_dataset() -> tuple[list[str], list[str]]:
    frame = pd.read_csv(DATA, usecols=["content", "sentiment"], encoding="utf-8-sig")
    frame = frame.dropna(subset=["content", "sentiment"])
    frame["content"] = frame["content"].astype(str).str.strip()
    frame = frame[frame["content"].str.len() > 0]
    frame = frame.drop_duplicates(subset=["content"]).reset_index(drop=True)
    texts = frame["content"].tolist()
    labels = [normalize_label(value) for value in frame["sentiment"]]
    return texts, labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    texts, labels = load_dataset()
    distribution = {label: labels.count(label) for label in sorted(set(labels))}
    print(f"dataset: {len(texts)} unik | {json.dumps(distribution)}", flush=True)

    preprocessor = TextPreprocessor()
    started = time.perf_counter()
    processed = []
    for index, text in enumerate(texts, start=1):
        processed.append(preprocessor.process(text))
        if index % 5000 == 0:
            print(f"  preprocessing {index}/{len(texts)}", flush=True)
    print(f"preprocessing selesai dalam {time.perf_counter() - started:.1f} s", flush=True)

    x_train, x_test, y_train, y_test = train_test_split(
        processed, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    summary = {}
    engines = (
        NaiveBayesEngine(max_features=MAX_FEATURES, min_df=2),
        SVMEngine(max_features=MAX_FEATURES, min_df=2),
    )
    for engine in engines:
        started = time.perf_counter()
        engine.train(x_train, y_train)
        elapsed = time.perf_counter() - started
        predicted = [prediction.label for prediction in engine.predict(x_test)]
        print(f"\n=== {engine.name} ({elapsed:.1f} s) ===")
        print(classification_report(y_test, predicted, digits=4, zero_division=0), flush=True)
        artifact = OUT / str(engine.pretrained_filename)
        joblib.dump(engine.state_dict(), artifact, compress=3)
        summary[engine.name] = {
            "train_seconds": round(elapsed, 2),
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 4),
            "file": artifact.name,
            "size_kb": round(artifact.stat().st_size / 1024, 1),
            "train_size": len(x_train),
            "test_size": len(x_test),
        }
        print(f"tersimpan: {artifact} ({summary[engine.name]['size_kb']} KB)", flush=True)

    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
