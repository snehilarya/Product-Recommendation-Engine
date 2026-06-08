import json
import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_products(path: str):
    """
    Load the LDJSON product file and return cleaned data structures.

    Returns:
        df: cleaned DataFrame with one row per product
        id_to_index: dict mapping product_id (str) -> row index (int)
        index_to_id: list mapping row index (int) -> product_id (str)
    """
    records = []
    failed = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                failed += 1

    total = len(records) + failed
    if failed:
        logger.warning(f"Skipped {failed}/{total} malformed records ({100*failed/total:.1f}%)")
    logger.info(f"Parsed {len(records)}/{total} records successfully")

    df = pd.DataFrame(records)
    df = _clean(df)

    id_to_index = {product_id: i for i, product_id in enumerate(df["uniq_id"])}
    index_to_id = df["uniq_id"].tolist()

    return df, id_to_index, index_to_id


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["uniq_id"])
    df = df.drop_duplicates(subset=["uniq_id"])

    df["sales_price"] = df["sales_price"].apply(_parse_price)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["weight"] = df["weight"].apply(_parse_weight)
    df["bestsellers_rank"] = df["product_details__k_v_pairs"].apply(_extract_rank)
    df["child_category"] = df["parent___child_category__all"].apply(_extract_child_category)

    # TF-IDF concatenation requires string — nulls become empty string
    text_columns = ["product_name", "brand", "colour", "other_items_customers_buy"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # Remove near-duplicate SKUs: same product name + price + brand (when brand is
    # known) is almost certainly the same item listed by multiple sellers.
    # We require brand to be non-empty before including it in the key — two products
    # with null brand, same name, and same price may be genuinely different items.
    before_dedup = len(df)
    df["_dedup_key"] = (
        df["product_name"].str.lower().str.strip() + "|" +
        df["sales_price"].astype(str) + "|" +
        df["brand"].apply(lambda b: b.lower().strip() if b else "__unknown__" + str(id(b)))
    )
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    removed = before_dedup - len(df)
    if removed:
        logger.info(f"Removed {removed} near-duplicate SKUs (same name+price+brand)")

    after = len(df)
    logger.info(f"Clean complete: {before} → {after} products")
    return df.reset_index(drop=True)


def _parse_price(value) -> Optional[float]:
    """Parse price string like '200.00' to float. Returns None if unparseable."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _parse_weight(value) -> Optional[float]:
    """
    Parse weight string like '86.2 g' to float. Returns None for the sentinel
    value 999999999 the dataset uses to indicate unknown weight.
    Only 21% of products have a real weight value; the rest are NaN after this.
    """
    if value is None:
        return None
    try:
        numeric_part = str(value).split()[0].replace(",", "")
        parsed = float(numeric_part)
        return None if parsed >= 999999999 else parsed
    except (ValueError, IndexError):
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
