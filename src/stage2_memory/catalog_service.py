"""
src/stage2_memory/catalog_service.py
───────────────────────────────────
Showroom Catalog Service:
Loads and caches the 25 certified vehicles from data/knowledge/cars_inventory.csv.
Formats a high-density, compact markdown inventory catalog (~480 tokens) for Llama 3.2.
Provides structured vehicle querying and mathematical budget audits without brittle regex.
"""

import os
import csv
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

log = logging.getLogger(__name__)

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "cars_inventory.csv"


class CatalogService:
    """Manages the certified vehicles inventory catalog."""

    def __init__(self, csv_file: Optional[Path] = None):
        self.csv_path = csv_file or CSV_PATH
        self.cars: List[Dict[str, Any]] = []
        self._cached_catalog_text: str = ""
        self.load_inventory()

    def load_inventory(self):
        """Loads and parses cars_inventory.csv."""
        if not self.csv_path.exists():
            log.warning("CatalogService: CSV file not found at %s", self.csv_path)
            return

        cars_list = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price_str = row.get("Price_NPR_Lakhs", "").strip()
                    # Parse numeric float for price (e.g. "15.5 Lakhs" -> 15.5)
                    price_float = 0.0
                    try:
                        clean_num = price_str.lower().replace("lakhs", "").replace("lakh", "").strip()
                        price_float = float(clean_num)
                    except ValueError:
                        pass

                    cars_list.append({
                        "stock_id": row.get("Stock_ID", "").strip(),
                        "brand": row.get("Brand", "").strip(),
                        "model": row.get("Model", "").strip(),
                        "variant": row.get("Variant", "").strip(),
                        "year": row.get("Make_Year", "").strip(),
                        "cc": row.get("Engine_CC", "").strip(),
                        "lot": row.get("Lot_Number", "").strip(),
                        "transmission": row.get("Transmission", "").strip(),
                        "fuel": row.get("Fuel_Type", "").strip(),
                        "mileage": row.get("Mileage_KM", "").strip(),
                        "price_str": price_str,
                        "price_float": price_float,
                        "condition": row.get("Condition", "").strip(),
                        "tax": row.get("Tax_Status", "").strip(),
                        "features": row.get("Features", "").strip(),
                    })
        except Exception as e:
            log.error("Failed to read inventory CSV: %s", e)
            return

        self.cars = cars_list
        self._cached_catalog_text = self._build_compact_catalog()
        log.info("CatalogService: loaded %d certified vehicles into catalog.", len(self.cars))

    def _build_compact_catalog(self) -> str:
        """Constructs a high-density tabular representation of all 25 vehicles (~480 tokens)."""
        lines = [
            "| Model & Variant | Year | Engine | Lot | Transmission | Price (NPR) | Highlights |",
            "|:---|:---:|:---:|:---:|:---|:---|:---|"
        ]
        for c in self.cars:
            car_name = f"{c['brand']} {c['model']} {c['variant']}".strip()
            lines.append(
                f"| {car_name} | {c['year']} | {c['cc']} | {c['lot']} | {c['transmission']} | NPR {c['price_str']} | {c['features'][:35]} |"
            )
        return "\n".join(lines)

    def get_catalog_prompt_text(self) -> str:
        """Returns the high-density inventory table for injection into system context."""
        if not self._cached_catalog_text:
            self.load_inventory()
        return self._cached_catalog_text

    def find_cars_by_price(self, max_price: float) -> List[Dict[str, Any]]:
        """Returns certified cars where price <= max_price, sorted closest to budget first."""
        matching = [c for c in self.cars if 0.0 < c["price_float"] <= max_price]
        matching.sort(key=lambda x: x["price_float"], reverse=True)
        return matching

    def get_budget_context_block(self, budget_text_or_num: Any) -> str:
        """
        Calculates mathematical budget audit without any hardcoded queries.
        Pre-sorts inventory into strictly within budget (Price <= B) and slightly above budget.
        """
        if not budget_text_or_num:
            return ""

        budget_float = 0.0
        if isinstance(budget_text_or_num, (int, float)):
            budget_float = float(budget_text_or_num)
        else:
            m = re.search(r"(\d+(?:\.\d+)?)", str(budget_text_or_num))
            if m:
                budget_float = float(m.group(1))

        if budget_float <= 0.0:
            return ""

        matching = [c for c in self.cars if 0.0 < c["price_float"] <= budget_float]
        matching.sort(key=lambda x: x["price_float"], reverse=True)

        above = [c for c in self.cars if budget_float < c["price_float"] <= budget_float + 4.5]
        above.sort(key=lambda x: x["price_float"])

        lines = [f"\n[MATHEMATICAL BUDGET AUDIT — CUSTOMER BUDGET: {budget_float} LAKHS]"]
        if matching:
            lines.append(f"• CERTIFIED CARS STRICTLY WITHIN BUDGET (Price <= {budget_float} Lakhs):")
            for c in matching:
                lines.append(f"  - {c['year']} {c['brand']} {c['model']} {c['variant']} ({c['cc']}, Lot {c['lot']}, {c['transmission']}) — NPR {c['price_str']}")
        else:
            lines.append(f"• (No certified cars currently in stock at or below NPR {budget_float} Lakhs)")

        if above:
            lines.append(f"• CARS SLIGHTLY ABOVE BUDGET (Exceeds customer budget):")
            for c in above[:3]:
                diff = round(c['price_float'] - budget_float, 1)
                lines.append(f"  - {c['year']} {c['brand']} {c['model']} {c['variant']} — NPR {c['price_str']} (+{diff} Lakhs above budget)")

        lines.append("STRICT INSTRUCTION: Only recommend cars from the 'STRICTLY WITHIN BUDGET' list. Never claim a car from the 'ABOVE BUDGET' list is within budget.")
        return "\n".join(lines)


catalog_service = CatalogService()
