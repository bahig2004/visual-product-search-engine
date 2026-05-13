import re

ALIASES = {
    "tshirts": "tshirt",
    "t-shirts": "tshirt",
    "t shirt": "tshirt",
    "tee": "tshirt",
    "tee shirt": "tshirt",
    "sneakers": "sneaker",
    "shoes": "shoe",
    "pants": "pant",
    "jeans": "jean",
}


def normalize_query(query: str) -> str:
    query = query.lower().strip()
    query = query.replace("-", " ")
    query = re.sub(r"[^a-z0-9\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    if query in ALIASES:
        return ALIASES[query]

    return query
