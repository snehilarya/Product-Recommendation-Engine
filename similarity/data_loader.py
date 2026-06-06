import json
import re
from typing import Optional

import pandas as pd


def load_products(path: str):
    """
    Load the LDJSON product file and return cleaned data structures.

    Returns:
        df: cleaned DataFrame with one row per product
        id_to_index: dict mapping product_id (str) -> row index (int)
        index_to_id: list mapping row index (int) -> product_id (str)
    """
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    df = _clean(df)

    id_to_index = {product_id: i for i, product_id in enumerate(df["uniq_id"])}
    index_to_id = df["uniq_id"].tolist()

    return df, id_to_index, index_to_id


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["uniq_id"])
    df = df.drop_duplicates(subset=["uniq_id"])

    df["weight"] = df["weight"].apply(_parse_weight)
    df["sales_price"] = df["sales_price"].apply(_parse_price)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["bestsellers_rank"] = df["product_details__k_v_pairs"].apply(_extract_rank)
    df["child_category"] = df["parent___child_category__all"].apply(_extract_child_category)

    # TF-IDF concatenation requires string — nulls become empty string
    text_columns = ["product_name", "brand", "colour", "other_items_customers_buy"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    return df.reset_index(drop=True)


def _parse_weight(value) -> Optional[float]:
    """
    Parse weight string. Returns None for the sentinel value 999999999
    which the dataset uses to indicate 'unknown weight'.
    """
    if value is None:
        return None
    try:
        # weight field can be "86.2 g" or "999999999" — take the numeric part
        numeric_part = str(value).split()[0].replace(",", "")
        parsed = float(numeric_part)
        if parsed >= 999999999:
            return None
        return parsed
    except (ValueError, IndexError):
        return None


def _parse_price(value) -> Optional[float]:
    """Parse price string like '200.00' to float. Returns None if unparseable."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _extract_rank(details) -> Optional[int]:
    """Pull the overall Amazon bestsellers rank from product_details dict."""
    if not isinstance(details, dict):
        return None
    raw = details.get("Amazon_Bestsellers_Rank", "")
    match = re.search(r"#([\d,]+)", str(raw))
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _extract_child_category(categories) -> Optional[str]:
    """Return the most-specific (last) key from the category dict."""
    if not isinstance(categories, dict) or not categories:
        return None
    return list(categories.keys())[-1]
