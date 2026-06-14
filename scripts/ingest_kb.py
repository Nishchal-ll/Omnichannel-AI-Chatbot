"""
scripts/ingest_kb.py
Run: python scripts/ingest_kb.py
Run with --force to reload after editing KB files.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge import ingest_kb, kb_stats

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Re-ingest even if KB exists")
args = parser.parse_args()

print("Ingesting knowledge base…")
count = ingest_kb(force=args.force)
stats = kb_stats()
print(f"Done. {count} chunks stored in {stats['db_path']}")
