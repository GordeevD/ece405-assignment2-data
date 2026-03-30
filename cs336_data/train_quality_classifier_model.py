from __future__ import annotations

from pathlib import Path

from cs336_data.quality_classifier import train_quality_classifier


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests" / "fixtures"
    train_path = root / "cs336_data" / "quality_train.txt"
    model_path = root / "cs336_data" / "quality_fasttext.bin"

    wiki_text = (fixtures / "high_quality_wiki_reference.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    cc_text = (fixtures / "low_quality_cc.txt").read_text(
        encoding="utf-8", errors="ignore"
    )

    wiki_lines = [
        line.strip() for line in wiki_text.split("\n") if len(line.strip().split()) >= 6
    ]
    cc_lines = [
        line.strip() for line in cc_text.split("\n") if len(line.strip()) >= 4
    ]

    extra_wiki = [
        "Peer reviewed reference article with definitions, context, citations, and coherent structure.",
        "Educational encyclopedia style writing with precise language and topic organization.",
        "Long form explanatory prose that introduces concepts and develops arguments carefully.",
        "Academic style summary with formal tone and connected paragraphs.",
        "Policy report discussing methods, assumptions, results, and limitations.",
        "Well edited documentation with complete sentences and clear terminology.",
        "Historical background section followed by analysis and interpretation.",
        "Technical tutorial explaining principles, examples, and tradeoffs in detail.",
        "Research overview with balanced discussion and neutral, informative style.",
    ]

    extra_cc = [
        "register register login forum memberlist profile usergroups search search",
        "click here sign up now free bonus limited offer buy now buy now",
        "powered by generic template all rights reserved contact admin",
        "cheap deals promo code promo code promo code limited time",
        "navigation menu home home home footer terms cookies privacy",
        "spam keywords repeated repeated repeated with little information",
        "short fragmented text with boilerplate and template artifacts",
        "affiliate links and seo stuffing without meaningful content",
        "auto generated page text with duplicated phrases and low value",
    ]

    examples: list[str] = []
    examples.extend(f"__label__wiki {line}" for line in wiki_lines)
    examples.extend(f"__label__cc {line}" for line in cc_lines)
    examples.extend(f"__label__wiki {line}" for line in extra_wiki)
    examples.extend(f"__label__cc {line}" for line in extra_cc)

    # Chunking improves sample count without introducing external data.
    wiki_words = wiki_text.split()
    for i in range(0, min(len(wiki_words), 8000), 80):
        segment = " ".join(wiki_words[i : i + 80]).strip()
        if len(segment.split()) >= 20:
            examples.append(f"__label__wiki {segment}")

    cc_words = cc_text.split()
    for i in range(0, min(len(cc_words), 3000), 35):
        segment = " ".join(cc_words[i : i + 35]).strip()
        if len(segment.split()) >= 8:
            examples.append(f"__label__cc {segment}")

    train_path.write_text("\n".join(examples) + "\n", encoding="utf-8")
    print(f"wrote training data: {train_path} ({len(examples)} lines)")

    out = train_quality_classifier(
        train_file=train_path,
        model_output_path=model_path,
        epoch=40,
        lr=0.5,
        word_ngrams=2,
        dim=100,
        bucket=50000,
    )
    print(f"model saved: {out}")
    print(f"model size bytes: {out.stat().st_size}")


if __name__ == "__main__":
    main()
