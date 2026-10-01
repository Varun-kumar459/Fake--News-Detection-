"""Create a small synthetic demo dataset (data/sample_news.csv) so the project
runs out of the box. For real accuracy, use the ISOT dataset (see README).

    python generate_sample_data.py
"""
import random

import pandas as pd

import config

random.seed(config.RANDOM_STATE)

SUBJECTS = ["The finance minister", "A federal court", "The central bank", "City officials",
            "Researchers at a university", "The health ministry", "The transport department",
            "A parliamentary committee", "The election commission", "Local police"]
REAL_ACTIONS = [
    "announced a revised budget after weeks of consultation with industry groups",
    "published a report showing a 2.3 percent rise in quarterly growth",
    "confirmed that the new policy will take effect next month, according to an official statement",
    "said the investigation is ongoing and results will be released after review",
    "approved funding for road repairs and public transport upgrades",
    "released data from a peer-reviewed study of 4,000 participants",
    "scheduled a public hearing to discuss the proposed regulations",
    "reported that unemployment fell slightly compared with the previous quarter",
]
REAL_TAILS = ["Officials said further details would be shared at a press briefing.",
              "The report was reviewed by independent experts.",
              "Spokespeople declined to comment beyond the written statement.",
              "The decision follows a vote held on Tuesday.",
              "Analysts said the figures were in line with expectations."]

FAKE_SUBJECTS = ["SHOCKING: Secret elites", "BREAKING!!! Celebrity doctor", "You won't believe what the government",
                 "Insiders reveal that politicians", "Anonymous source EXPOSES how big pharma",
                 "Whistleblower claims the media", "THEY don't want you to know that scientists"]
FAKE_ACTIONS = [
    "are hiding the TRUTH about a miracle cure that doctors hate",
    "secretly control the weather to destroy your freedom, share before it is deleted",
    "faked the whole event and the mainstream media is lying to you again",
    "planned a massive cover up, wake up people, this is 100% proven",
    "put mind control chemicals in tap water, forward this to everyone you know",
    "admitted everything in a leaked video that was instantly banned",
    "will arrest millions overnight, the shocking proof they tried to erase",
    "found a one weird trick to get rich fast, banks are furious",
]
FAKE_TAILS = ["Share this now before they delete it!!!", "Is the media lying to you? Wake up!",
              "Click here to see the banned video!", "This changes EVERYTHING, spread the word!",
              "Do your own research, they are watching you!"]


def make(subjects, actions, tails, label, n):
    rows = []
    for _ in range(n):
        text = f"{random.choice(subjects)} {random.choice(actions)}. {random.choice(tails)}"
        rows.append({"text": text, "label": label})
    return rows


def main():
    rows = make(SUBJECTS, REAL_ACTIONS, REAL_TAILS, 0, 400) + make(FAKE_SUBJECTS, FAKE_ACTIONS, FAKE_TAILS, 1, 400)
    df = pd.DataFrame(rows).sample(frac=1, random_state=config.RANDOM_STATE)
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.SAMPLE_CSV, index=False)
    print(f"Wrote {len(df)} rows to {config.SAMPLE_CSV}")


if __name__ == "__main__":
    main()
