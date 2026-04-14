from think.indexer.journal import search_journal
import json

# The user's query parameters
query = "" # Optional: add a keyword search
day = "20260414"
facet = "ping_identity"

total_hits, results = search_journal(query, day=day, facet=facet, limit=100)

print(f"Found {total_hits} results.")
for result in results:
    print(json.dumps(result, indent=2))
