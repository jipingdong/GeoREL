# GeoREL

**Geocoded Records of Events and Locations from Global News**

GeoREL is an article-level event and location dataset constructed from news linked in the Global Database of Events, Language, and Tone (GDELT). This repository contains the production code for retrieving source webpages, extracting event information with large language models, and adding temporal and geographic information.

Related manuscript: *An article-level event dataset from global news with spatiotemporal enrichment*.

## Main Features

- Groups GDELT records by source URL, retains linked event identifiers and retrieves webpage HTML and titles.
- Screens articles for event relevance and extracts summaries, actors, individuals, event categories, sentiment, importance and Goldstein-style scores.
- Extracts article text from local HTML, event time and role-aware locations, geocodes locations, and normalizes country names and ISO3 codes.
- Deduplicates articles by URL, combines associated GDELT identifiers and exports the dataset with processing reports.

The two components are documented separately:

| Directory | Purpose | Execution guide |
| --- | --- | --- |
| `data_crawl/` | Retrieve webpages from local GDELT export archives | [Article retrieval](data_crawl/readme.md) |
| `data_process/` | Seven steps from local HTML text extraction to dataset export | [Data processing](data_process/readme.md) |

## Installation

Python 3.11 or later is required. Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The GeoREL release contains **550,574 URL-unique article records**, **15 fields** and **9,151 source hostnames**, with GDELT collection times from January through June 2024. The data package (`georel_dataset.csv.gz`) is published on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.23662152
- URL: https://zenodo.org/records/23662152
