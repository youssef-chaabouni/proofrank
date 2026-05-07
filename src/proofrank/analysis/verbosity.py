import re
import tiktoken
import spacy
import matplotlib.pyplot as plt
import numpy as np
import signal

nlp = spacy.load(
    "en_core_web_md",
    disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
)
enc = tiktoken.get_encoding("o200k_base")
MAX_CHARS_FOR_SPACY = 20_000


class TimeoutException(Exception):
    pass

def _handle_timeout(signum, frame):
    raise TimeoutException("calculate_length timed out")

def calculate_length_timeout(response_text, timeout_sec=2):
    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_sec)
    try:
        return calculate_length(response_text)
    except TimeoutException:
        print(f"[TIMEOUT] text len={len(str(response_text))}")
        return (None,) * 7
    except Exception as e:
        print(f"[ERROR] calculate_length failed: {e}")
        return (None,) * 7
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def calculate_length(response_text):
    char_count_nws = len(re.sub(r"\s+", "", response_text))

    word_count = len(response_text.split())

    standard_token_count = len(enc.encode(response_text))
    english_words, non_english_words = extract_proper_english(response_text)

    if len(response_text) > MAX_CHARS_FOR_SPACY:
        return None, None, None, None, None, None, None

    return (
        char_count_nws,
        word_count,
        standard_token_count,
        len(english_words),
        sum(len(w) for w in english_words),
        len(non_english_words),
        sum(len(w) for w in non_english_words),
    )


def extract_proper_english(text):
    doc = nlp(text)

    english_words = []
    non_english_words = []

    for token in doc:
        if not token.is_alpha:
            non_english_words.append(token.text)
            continue

        if len(token.text) == 1: #and token.text.lower() not in ["a", "i"]:
            non_english_words.append(token.text)
            continue

        if token.is_oov:
            non_english_words.append(token.text)
            continue

        english_words.append(token.text)

    return english_words, non_english_words


def print_verbosity_stats(df):
    sol_lengths = df["solution_rephrase"].apply(calculate_length)
    df.loc[:, "solution_spacy_word_count"] = sol_lengths.apply(lambda x: x[3])

    df = df[df["solution_char_count"] > 0].copy()

    rephrase_lengths = df["outputs_rephrase"].apply(
        lambda outputs: (
            [calculate_length(output) for output in outputs]
            if isinstance(outputs, list)
            else [calculate_length(outputs)]
        )
    )

    df.loc[:, "rephrase_spacy_word_counts"] = rephrase_lengths.apply(
        lambda x: sum([length[3] for length in x]) / len(x)
    )

    df = df[
        (df["solution_char_count"] / df["rephrase_char_counts"] > 0.02)
        & (df["solution_char_count"] / df["rephrase_char_counts"] < 20)
    ].copy()

    print("Solution Length Stats:")
    judge_stats = (
        df.groupby(["solver_id"])
        .agg(
            solution_spacy_word_count=("solution_spacy_word_count", "mean"),
            rephrase_spacy_word_counts=("rephrase_spacy_word_counts", "mean"),

        )
        .reset_index()
    )

    judge_stats["spacy_word_count_ratio"] = (
        judge_stats["solution_spacy_word_count"]
        / judge_stats["rephrase_spacy_word_counts"]
    )

    print(judge_stats.to_string(index=False))

    count_types = [
        ("Spacy Word Count", "solution_spacy_word_count", "rephrase_spacy_word_counts"),
    ]

    solvers = df["solver_id"].unique()

    for solver in solvers:
        solver_df = df[df["solver_id"] == solver]

        for name, orig_col, rep_col in count_types:
            ratios = solver_df[orig_col] / solver_df[rep_col].replace(0, float("nan"))
            ratios = ratios.dropna()
            ratios = ratios[
                (ratios < 20) & (ratios > 0.02)
            ]  # Filter out extreme values

            plt.figure(figsize=(8, 6))
            plt.hist(
                ratios,
                bins=int(max(ratios)) * 10,
                alpha=0.7,
                edgecolor="black",
                color="skyblue",
            )

            plt.title(f"{solver}: {name} Ratio Distribution\n(Original / Rephrased)")
            plt.xlabel(f"Ratio (Undefined if denominator is 0)")
            plt.ylabel("Frequency")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            plt.axvline(
                np.mean(ratios),
                color="tab:red",
                linestyle="dashed",
                linewidth=1,
                label="Mean",
            )
            plt.axvline(
                np.median(ratios),
                color="tab:green",
                linestyle="dashed",
                linewidth=1,
                label="Median",
            )

            plt.legend()

            plt.savefig(
                f"figures/{solver.replace('/', '_')}_{name.replace(' ', '_').lower()}_ratio_histogram.png"
            )

    for name, orig_col, rep_col in count_types:

        plt.figure(figsize=(8, 6))

        for solver in solvers:
            solver_df = df[df["solver_id"] == solver]
            ratios = solver_df[orig_col] / solver_df[rep_col].replace(0, float("nan"))
            ratios = ratios.dropna()
            ratios = ratios[
                (ratios < 20) & (ratios > 0.02)
            ]  # Filter out extreme values
            plt.hist(
                ratios,
                bins=int(max(ratios)) * 10,
                alpha=0.7,
                edgecolor="black",
                label=solver,
                density=True,
            )

        plt.title(f"{solver}: {name} Ratio Distribution\n(Original / Rephrased)")
        plt.xlabel(f"Ratio (Undefined if denominator is 0)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.axvline(
            np.mean(ratios),
            color="tab:red",
            linestyle="dashed",
            linewidth=1,
            label="Mean",
        )
        plt.axvline(
            np.median(ratios),
            color="tab:green",
            linestyle="dashed",
            linewidth=1,
            label="Median",
        )

        plt.legend()

        plt.savefig(f"figures/{name.replace(' ', '_').lower()}_ratio_histogram.png")
    return df
