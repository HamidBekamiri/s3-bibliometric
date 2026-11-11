"""Generate a small synthetic Scopus-like CSV (3 topics) for smoke tests."""
import random, pandas as pd
random.seed(0)
topics = {
 "open innovation": ["open innovation", "external knowledge", "R&D collaboration", "absorptive capacity", "innovation networks"],
 "mergers and acquisitions": ["mergers and acquisitions", "acquirer returns", "deal premiums", "post-merger integration", "takeover bids"],
 "country of origin": ["country-of-origin effects", "consumer ethnocentrism", "product evaluations", "brand origin", "purchase intention"],
}
rows, i = [], 0
for t, terms in topics.items():
    for k in range(20):
        a, b, c = random.sample(terms, 3)
        y = random.randint(2005, 2023)
        title = f"{a.capitalize()} and {b}: evidence from {random.choice(['European','US','Asian','Nordic'])} firms ({k})"
        abstract = (f"This study examines how {a} relates to {b} in the context of {t}. "
                    f"Drawing on data from {random.randint(80,900)} firms, we find that {c} shapes the link between {a} and {b}. "
                    f"We discuss implications of {t} research for managers and theory on {c}.")
        rows.append(dict(Title=title, Abstract=abstract, Year=y, **{"Cited by": random.randint(0, 300)},
                         Authors=f"Author{random.randint(1,25)} A.; Author{random.randint(1,25)} B.",
                         **{"Author Keywords": "; ".join([a, b, c])}, EID=f"2-s2.0-{100000+i}",
                         **{"Source title": random.choice(["Research Policy", "SMJ", "JIBS", "JBR"])}))
        i += 1
rows.append(dict(Title="Photosynthesis rates in alpine plants", Abstract="We measure leaf-level photosynthesis in alpine meadows.", Year=2015, **{"Cited by": 3}, Authors="Botanist C.", **{"Author Keywords": "photosynthesis"}, EID="2-s2.0-999999", **{"Source title": "Ecology"}))
df = pd.DataFrame(rows)
# References: each paper cites 3 earlier same-topic papers by title (Scopus style)
refs = []
for j, r in df.iterrows():
    base = (j // 20) * 20
    cands = [df.at[m, "Title"] for m in range(base, min(base + 20, len(df) - 1)) if m != j]
    picks = random.sample(cands, 3) if j < len(df) - 1 else []
    if j < len(df) - 1 and random.random() < 0.3:
        picks.append(df.at[random.randrange(0, len(df) - 1), "Title"])  # cross-topic
    refs.append("; ".join(f"Smith J., {p}, Journal, 2010" for p in picks))
df["References"] = refs
df.to_csv("examples/sample_scopus.csv", index=False)
print(df.shape)
