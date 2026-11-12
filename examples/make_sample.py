"""Generate a synthetic Scopus-like CSV for demos and tests.

4 business-research topics (one of them sits between two others), with authors,
affiliations, journals, funding, author keywords and Scopus-style reference lists.
All content is invented.
"""
import random
from pathlib import Path

import pandas as pd

random.seed(1)

TOPICS = {
    "open innovation": {
        "terms": ["open innovation", "external knowledge sourcing", "R&D collaboration", "absorptive capacity",
                  "innovation networks", "crowdsourcing ideas", "university-industry links", "inbound innovation"],
        "years": (2005, 2023), "journals": ["Research Policy", "Technovation", "R&D Management"],
        "countries": ["Germany", "Italy", "Denmark", "Netherlands"], "funder": "European Commission",
    },
    "mergers and acquisitions": {
        "terms": ["mergers and acquisitions", "acquirer abnormal returns", "deal premiums", "post-merger integration",
                  "takeover defenses", "target selection", "cross-border deals", "acquisition experience"],
        "years": (2003, 2018), "journals": ["Strategic Management Journal", "Journal of Finance", "Journal of Corporate Finance"],
        "countries": ["United States", "United Kingdom", "Canada"], "funder": "National Science Foundation",
    },
    "country of origin": {
        "terms": ["country-of-origin effects", "consumer ethnocentrism", "product evaluations", "brand origin",
                  "purchase intention", "made-in labels", "country image", "animosity toward foreign products"],
        "years": (2004, 2021), "journals": ["Journal of International Business Studies", "International Marketing Review", "Journal of Business Research"],
        "countries": ["China", "South Korea", "United States", "Spain"], "funder": "National Natural Science Foundation of China",
    },
    "technology acquisitions": {  # bridges open innovation and M&A
        "terms": ["technology acquisitions", "acquiring startups for innovation", "external knowledge sourcing",
                  "post-merger integration", "absorptive capacity", "acquisition of R&D capabilities",
                  "innovation output after acquisitions", "acqui-hiring"],
        "years": (2014, 2024), "journals": ["Research Policy", "Strategic Management Journal", "Technovation"],
        "countries": ["United States", "Germany", "Denmark"], "funder": "Danish Independent Research Fund",
    },
}
UNIS = {"Germany": "Technical University of Munich, Munich", "Italy": "Politecnico di Milano, Milan",
        "Denmark": "Aalborg University, Aalborg", "Netherlands": "Erasmus University Rotterdam, Rotterdam",
        "United States": "Harvard Business School, Boston", "United Kingdom": "London Business School, London",
        "Canada": "University of Toronto, Toronto", "China": "Peking University, Beijing",
        "South Korea": "Seoul National University, Seoul", "Spain": "IE Business School, Madrid"}
SIZES = {"open innovation": 45, "mergers and acquisitions": 40, "country of origin": 40, "technology acquisitions": 25}

rows = []
authors_by_topic = {t: [f"{n} {chr(65 + k % 26)}." for k, n in enumerate(random.sample(
    ["Jensen", "Rossi", "Smith", "Chen", "Kim", "Garcia", "Muller", "Brown", "Wang", "Larsen", "Bianchi",
     "Taylor", "Li", "Park", "Lopez", "Schmidt", "Nielsen", "Davis", "Zhang", "Moreno"], 9))] for t in TOPICS}
# brokers: shared authors
authors_by_topic["technology acquisitions"] += authors_by_topic["open innovation"][:3] + authors_by_topic["mergers and acquisitions"][:3]
authors_by_topic["country of origin"] += authors_by_topic["mergers and acquisitions"][3:4]

for topic, spec in TOPICS.items():
    for k in range(SIZES[topic]):
        a, b, c = random.sample(spec["terms"], 3)
        lo, hi = spec["years"]
        y = int(min(hi, lo + abs(random.gauss(0, 1)) * (hi - lo) / 1.6)) if topic != "mergers and acquisitions" \
            else int(max(lo, hi - abs(random.gauss(0, 1)) * (hi - lo) / 1.6))
        country = random.choice(spec["countries"])
        title = f"{a.capitalize()} and {b}: evidence from {random.choice(['European', 'US', 'Asian', 'Nordic', 'global'])} firms"
        if any(r["Title"] == title for r in rows):
            title += f" ({y})"
        abstract = (f"This study examines how {a} relates to {b} in research on {topic}. "
                    f"Using data on {random.randint(80, 900)} firms, we show that {c} shapes the link between {a} and {b}. "
                    f"We discuss implications for theory on {topic} and for managers dealing with {c}.")
        au = random.sample(authors_by_topic[topic], random.randint(1, 3))
        aff = "; ".join(f"Department of Management, {UNIS[random.choice(spec['countries'])]}, "
                        f"{random.choice(spec['countries'])}" for _ in au)
        aff = aff.replace("Department of Management, " + UNIS[country].split(',')[0], "Department of Management, " + UNIS[country].split(',')[0])
        rows.append({
            "Authors": "; ".join(au), "Title": title, "Year": y,
            "Source title": random.choice(spec["journals"]),
            "Cited by": int(random.paretovariate(1.3) * 8 * max(1, 2025 - y) ** 0.6),
            "Affiliations": aff, "Abstract": abstract, "Author Keywords": "; ".join([a, b, c]),
            "Funding Details": spec["funder"] if random.random() < 0.45 else "",
            "EID": f"2-s2.0-{850000000 + len(rows)}", "_topic": topic,
        })

df = pd.DataFrame(rows)
# references: classic external works per topic + earlier corpus papers (same topic, sometimes other)
classics = {t: [f"{chr(65 + i)}uthor{i} X., Classic study {i} on {t}, (19{90 + i}) Classic Journal, {i}, pp. 1-20"
                for i in range(6)] for t in TOPICS}
bridge = {"technology acquisitions": ["open innovation", "mergers and acquisitions"]}
refs = []
for i, r in df.iterrows():
    t = r["_topic"]
    pool_t = [t] + bridge.get(t, [])
    ext = random.sample(classics[t], 3) + ([random.choice(classics[random.choice(pool_t)])] if t in bridge else [])
    earlier = df[(df["Year"] < r["Year"]) & (df["_topic"].isin(pool_t))]
    cites = earlier.sample(min(len(earlier), random.randint(1, 4)), random_state=i)["Title"].tolist() if len(earlier) else []
    if random.random() < 0.15:
        other = df[(df["Year"] < r["Year"]) & (df["_topic"] != t)]
        if len(other):
            cites.append(other.sample(1, random_state=i)["Title"].iat[0])
    refs.append("; ".join(ext + [f"Someone Y., {c}, (2010) Some Journal" for c in cites]))
df["References"] = refs

df = pd.concat([df, pd.DataFrame([{
    "Authors": "Botanist C.", "Title": "Photosynthesis rates in alpine plants", "Year": 2015,
    "Source title": "Ecology", "Cited by": 3, "Affiliations": "Dept of Biology, University of Bern, Bern, Switzerland",
    "Abstract": "We measure leaf-level photosynthesis in alpine meadows across an elevation gradient.",
    "Author Keywords": "photosynthesis; alpine plants", "Funding Details": "", "EID": "2-s2.0-999999999",
    "References": "Körner C., Alpine plant life, (2003) Springer"}])], ignore_index=True)
df = df.drop(columns="_topic").sample(frac=1, random_state=3).reset_index(drop=True)
out = Path(__file__).with_name("sample_scopus.csv")
df.to_csv(out, index=False)
print(out, df.shape)
