# GeoREL Data Processing

This directory implements the GeoREL dataset production pipeline: taking news article text as input, it performs text extraction, event-relevance screening, LLM-based semantic extraction, spatiotemporal extraction, geocoding and country-name normalization, and finally exports an article-level event-and-location dataset with 15 fields.

## Configure Once, Run Each Step Separately

Edit `config.json` (create it from the example below if it does not exist).

```json
{
  "paths": {
    "input_dir": "<crawler-data-dir>",
    "output_dir": "<working-output-dir>"
  },
  "batch_range": {
    "start": "<start-timestamp>",
    "end": "<end-timestamp>"
  },
  "llm": {
    "api_key": "",
    "base_url": "<model-api-url>",
    "model": "<model-name>",
    "workers": 10,
    "attempts": 3,
    "timeout": 180,
    "retry_delay": 1
  },
  "steps": {
    "1_enrich_urls_title": {"workers": 4},
    "2_relevance_filter": {},
    "3_semantic_extraction": {},
    "4_spatiotemporal_extraction": {},
    "5_geocode_locations": {"provider": "cache-only"},
    "6_normalize_countries": {"llm": true},
    "7_finalize_dataset": {}
  }
}
```

- `paths.input_dir`: crawler data directory containing timestamp subfolders.
- `paths.output_dir`: working directory for intermediate results and the final dataset.
- `batch_range`: start/end timestamps (14 digits); it selects crawler batches.
- `llm`: model API URL, key and global concurrency.

Seven independent commands. Run only the step you need and check the result before moving to the next:

```powershell
python enrich_urls_title.py
python relevance_filter.py
python semantic_extraction.py
python spatiotemporal_extraction.py
python geocode_locations.py
python normalize_countries.py
python finalize_dataset.py
```

## The Seven Steps

| Step | Purpose | Output |
| --- | --- | --- |
| 1 `enrich_urls_title` | Extract article text from local HTML, updating `urls_title.csv` in place | `<input_dir>/<timestamp>/urls_title.csv` |
| 2 `relevance_filter` | Screen for event relevance and create `TITLE`/`RAWTEXT` working copies | `02_relevance/` |
| 3 `semantic_extraction` | Use the LLM to extract summary, category, actors, individuals, sentiment, importance and a Goldstein-style score | `03_semantic/` |
| 4 `spatiotemporal_extraction` | Extract event time and role-aware location objects | `04_spacetime/` |
| 5 `geocode_locations` | Geocode locations | `05_geocoded/` |
| 6 `normalize_countries` | Normalize country names and ISO3 codes | `06_countries/` |
| 7 `finalize_dataset` | Deduplicate by URL, merge GDELT IDs and export the public table | `07_release/georel_dataset.csv` |

The dataset contains 15 columns: `event_id, url, title, summary, state_actors, non_state_actors, individuals, goldstein_scale_score, event_importance, event_sentiment, event_category, domain, collection_time, event_time, locations`.