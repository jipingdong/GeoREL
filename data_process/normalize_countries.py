"""Step 6: normalize country names and codes across enriched article fields.

Input:
    Geocoded article CSV/CSV.GZ with location and entity JSON.
    pycountry resolves exact current ISO names
    and codes; the inline preferred-name/ISO3 catalog constrains public names,
    the model prompt and response validation.
Output:
    normalized_articles.csv for final export, unresolved_countries.csv and
    run_summary.json. Location countries, state-actor names and individuals'
    country attributes are normalized without dropping articles or changing
    event times, actor roles and unrelated fields.
Execution:
    Exact pycountry/catalog lookup runs first. Config llm=true or --llm enables
    semantic classification of remaining distinct mentions with deepseek-flash (temperature 1),
    a parameter/hidden API key, bounded retries and fingerprint-checked caches.
    --dry-run requires no model request. Ambiguous mentions remain available
    for review rather than being guessed or silently discarded.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import contextlib
import copy
import csv
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
import re
import tempfile
import time
import unicodedata
from urllib.parse import urlsplit

import pycountry
from tqdm import tqdm


from data_process.utils.process_config import StepArgumentParser, guard_step_selection
from data_process.utils.llm_runtime import json_hash, make_client as _make_client, safe_error, write_json


LOCATION_FIELDS = ("locations", "locations_json", "geo_locations_json", "event_locations")
STATUS_FIELDS = ("country_normalization_status", "country_normalization_error")
SUCCESS = {"ok", "multi_country", "non_country"}
# Frozen preferred names followed by legacy aliases retained for provenance.
# Only exact pycountry names/codes and preferred catalog labels match directly.
# Covers all 249 ISO countries/territories and the existing user-assigned XKX
# compatibility entry. Legacy SYSTEM_PROMPT contributes names only, not policy.
COUNTRY_CATALOG = {
    "ABW": ("Aruba", "AW", "533",),
    "AFG": ("Afghanistan", "AF", "Islamic Republic of Afghanistan", "004",),
    "AGO": ("Angola", "AO", "Republic of Angola", "024",),
    "AIA": ("Anguilla", "AI", "660",),
    "ALA": ("\u00c5land Islands", "AX", "248",),
    "ALB": ("Albania", "AL", "Republic of Albania", "008",),
    "AND": ("Andorra", "AD", "Principality of Andorra", "020",),
    "ARE": ("United Arab Emirates", "AE", "784",),
    "ARG": ("Argentina", "AR", "Argentine Republic", "032",),
    "ARM": ("Armenia", "AM", "Republic of Armenia", "051",),
    "ASM": ("American Samoa", "AS", "016",),
    "ATA": ("Antarctica", "AQ", "010",),
    "ATF": ("French Southern Territories", "TF", "Territory of the French Southern and Antarctic Lands", "260",),
    "ATG": ("Antigua and Barbuda", "AG", "028",),
    "AUS": ("Australia", "AU", "Commonwealth of Australia", "036",),
    "AUT": ("Austria", "AT", "Republic of Austria", "040",),
    "AZE": ("Azerbaijan", "AZ", "Republic of Azerbaijan", "031",),
    "BDI": ("Burundi", "BI", "Republic of Burundi", "108",),
    "BEL": ("Belgium", "BE", "Kingdom of Belgium", "056",),
    "BEN": ("Benin", "BJ", "Republic of Benin", "204",),
    "BES": ("Bonaire, Saint Eustatius and Saba", "BQ", "Bonaire, Sint Eustatius and Saba", "535",),
    "BFA": ("Burkina Faso", "BF", "854",),
    "BGD": ("Bangladesh", "BD", "People's Republic of Bangladesh", "050",),
    "BGR": ("Bulgaria", "BG", "Republic of Bulgaria", "100",),
    "BHR": ("Bahrain", "BH", "Kingdom of Bahrain", "048",),
    "BHS": ("Bahamas", "BS", "Commonwealth of the Bahamas", "044",),
    "BIH": ("Bosnia and Herzegovina", "BA", "Republic of Bosnia and Herzegovina", "070",),
    "BLM": ("St. Barths", "BL", "Saint Barth\u00e9lemy", "Territorial collectivity of Saint-Barth\u00e9lemy", "652",),
    "BLR": ("Belarus", "BY", "Republic of Belarus", "112",),
    "BLZ": ("Belize", "BZ", "084",),
    "BMU": ("Bermuda", "BM", "060",),
    "BOL": ("Bolivia", "BO", "Bolivia, Plurinational State of", "Plurinational State of Bolivia", "068",),
    "BRA": ("Brazil", "BR", "Federative Republic of Brazil", "076",),
    "BRB": ("Barbados", "BB", "052",),
    "BRN": ("Brunei Darussalam", "BN", "Nation of Brunei, Abode of Peace", "096",),
    "BTN": ("Bhutan", "BT", "Kingdom of Bhutan", "064",),
    "BVT": ("Bouvet Island", "BV", "074",),
    "BWA": ("Botswana", "BW", "Republic of Botswana", "072",),
    "CAF": ("Central African Republic", "CF", "140",),
    "CAN": ("Canada", "CA", "124",),
    "CCK": ("Cocos (Keeling) Islands", "CC", "Territory of the Cocos (Keeling) Islands", "166",),
    "CHE": ("Switzerland", "CH", "Swiss Confederation", "756",),
    "CHL": ("Chile", "CL", "Republic of Chile", "152",),
    "CHN": ("China", "CN", "People's Republic of China", "156",),
    "CIV": ("C\u00f4te d'Ivoire", "CI", "Republic of C\u00f4te d'Ivoire", "384",),
    "CMR": ("Cameroon", "CM", "Republic of Cameroon", "120",),
    "COD": ("DR Congo", "CD", "Congo, The Democratic Republic of the", "Democratic Republic of the Congo", "drc", "180",),
    "COG": ("Congo Republic", "CG", "Congo", "Republic of the Congo", "178",),
    "COK": ("Cook Islands", "CK", "184",),
    "COL": ("Colombia", "CO", "Republic of Colombia", "170",),
    "COM": ("Comoros", "KM", "Union of the Comoros", "174",),
    "CPV": ("Cabo Verde", "CV", "Republic of Cabo Verde", "132",),
    "CRI": ("Costa Rica", "CR", "Republic of Costa Rica", "188",),
    "CUB": ("Cuba", "CU", "Republic of Cuba", "192",),
    "CUW": ("Cura\u00e7ao", "CW", "Country of Cura\u00e7ao", "531",),
    "CXR": ("Christmas Island", "CX", "162",),
    "CYM": ("Cayman Islands", "KY", "136",),
    "CYP": ("Cyprus", "CY", "Republic of Cyprus", "196",),
    "CZE": ("Czechia", "CZ", "Czech Republic", "203",),
    "DEU": ("Germany", "DE", "Federal Republic of Germany", "276",),
    "DJI": ("Djibouti", "DJ", "Republic of Djibouti", "262",),
    "DMA": ("Dominica", "DM", "Commonwealth of Dominica", "212",),
    "DNK": ("Denmark", "DK", "Kingdom of Denmark", "208",),
    "DOM": ("Dominican Republic", "DO", "214",),
    "DZA": ("Algeria", "DZ", "People's Democratic Republic of Algeria", "012",),
    "ECU": ("Ecuador", "EC", "Republic of Ecuador", "218",),
    "EGY": ("Egypt", "EG", "Arab Republic of Egypt", "818",),
    "ERI": ("Eritrea", "ER", "the State of Eritrea", "State of Eritrea", "232",),
    "ESH": ("Western Sahara", "EH", "732",),
    "ESP": ("Spain", "ES", "Kingdom of Spain", "724",),
    "EST": ("Estonia", "EE", "Republic of Estonia", "233",),
    "ETH": ("Ethiopia", "ET", "Federal Democratic Republic of Ethiopia", "231",),
    "FIN": ("Finland", "FI", "Republic of Finland", "246",),
    "FJI": ("Fiji", "FJ", "Republic of Fiji", "242",),
    "FLK": ("Falkland Islands", "FK", "Falkland Islands (Malvinas)", "238",),
    "FRA": ("France", "FR", "French Republic", "250",),
    "FRO": ("Faroe Islands", "FO", "234",),
    "FSM": ("Micronesia, Fed. Sts.", "FM", "Micronesia, Federated States of", "Federated States of Micronesia", "583",),
    "GAB": ("Gabon", "GA", "Gabonese Republic", "266",),
    "GBR": ("United Kingdom", "GB", "United Kingdom of Great Britain and Northern Ireland", "uk", "u k", "826",),
    "GEO": ("Georgia", "GE", "268",),
    "GGY": ("Guernsey", "GG", "831",),
    "GHA": ("Ghana", "GH", "Republic of Ghana", "288",),
    "GIB": ("Gibraltar", "GI", "292",),
    "GIN": ("Guinea", "GN", "Republic of Guinea", "324",),
    "GLP": ("Guadeloupe", "GP", "312",),
    "GMB": ("Gambia", "GM", "Republic of the Gambia", "270",),
    "GNB": ("Guinea-Bissau", "GW", "Republic of Guinea-Bissau", "624",),
    "GNQ": ("Equatorial Guinea", "GQ", "Republic of Equatorial Guinea", "226",),
    "GRC": ("Greece", "GR", "Hellenic Republic", "300",),
    "GRD": ("Grenada", "GD", "308",),
    "GRL": ("Greenland", "GL", "304",),
    "GTM": ("Guatemala", "GT", "Republic of Guatemala", "320",),
    "GUF": ("French Guiana", "GF", "Guiana", "254",),
    "GUM": ("Guam", "GU", "316",),
    "GUY": ("Guyana", "GY", "Republic of Guyana", "Co-operative Republic of Guyana", "328",),
    "HKG": ("Hong Kong", "HK", "Hong Kong Special Administrative Region of China", "Hong Kong SAR", "China, Hong Kong Special Administrative Region", "344",),
    "HMD": ("Heard and McDonald Islands", "HM", "Heard Island and McDonald Islands", "Territory of Heard Island and McDonald Islands", "334",),
    "HND": ("Honduras", "HN", "Republic of Honduras", "340",),
    "HRV": ("Croatia", "HR", "Republic of Croatia", "191",),
    "HTI": ("Haiti", "HT", "Republic of Haiti", "332",),
    "HUN": ("Hungary", "HU", "Republic of Hungary", "348",),
    "IDN": ("Indonesia", "ID", "Republic of Indonesia", "360",),
    "IMN": ("Isle of Man", "IM", "833",),
    "IND": ("India", "IN", "Republic of India", "356",),
    "IOT": ("British Indian Ocean Territory", "IO", "086",),
    "IRL": ("Ireland", "IE", "372",),
    "IRN": ("Iran", "IR", "Iran, Islamic Republic of", "Islamic Republic of Iran", "364",),
    "IRQ": ("Iraq", "IQ", "Republic of Iraq", "368",),
    "ISL": ("Iceland", "IS", "Republic of Iceland", "352",),
    "ISR": ("Israel", "IL", "State of Israel", "376",),
    "ITA": ("Italy", "IT", "Italian Republic", "380",),
    "JAM": ("Jamaica", "JM", "388",),
    "JEY": ("Jersey", "JE", "832",),
    "JOR": ("Jordan", "JO", "Hashemite Kingdom of Jordan", "400",),
    "JPN": ("Japan", "JP", "392",),
    "KAZ": ("Kazakhstan", "KZ", "Republic of Kazakhstan", "398",),
    "KEN": ("Kenya", "KE", "Republic of Kenya", "404",),
    "KGZ": ("Kyrgyzstan", "KG", "Kyrgyz Republic", "417",),
    "KHM": ("Cambodia", "KH", "Kingdom of Cambodia", "116",),
    "KIR": ("Kiribati", "KI", "Republic of Kiribati", "296",),
    "KNA": ("St. Kitts and Nevis", "KN", "Saint Kitts and Nevis", "659",),
    "KOR": ("South Korea", "KR", "Korea, Republic of", "Republic of Korea", "410",),
    "KWT": ("Kuwait", "KW", "State of Kuwait", "414",),
    "LAO": ("Laos", "LA", "Lao People's Democratic Republic", "418",),
    "LBN": ("Lebanon", "LB", "Lebanese Republic", "422",),
    "LBR": ("Liberia", "LR", "Republic of Liberia", "430",),
    "LBY": ("Libya", "LY", "State of Libya", "434",),
    "LCA": ("St. Lucia", "LC", "Saint Lucia", "662",),
    "LIE": ("Liechtenstein", "LI", "Principality of Liechtenstein", "438",),
    "LKA": ("Sri Lanka", "LK", "Democratic Socialist Republic of Sri Lanka", "144",),
    "LSO": ("Lesotho", "LS", "Kingdom of Lesotho", "426",),
    "LTU": ("Lithuania", "LT", "Republic of Lithuania", "440",),
    "LUX": ("Luxembourg", "LU", "Grand Duchy of Luxembourg", "442",),
    "LVA": ("Latvia", "LV", "Republic of Latvia", "428",),
    "MAC": ("Macau", "MO", "Macao", "Macao Special Administrative Region of China", "Macau SAR", "China, Macao Special Administrative Region", "446",),
    "MAF": ("Saint-Martin", "MF", "Saint Martin (French part)", "663",),
    "MAR": ("Morocco", "MA", "Kingdom of Morocco", "504",),
    "MCO": ("Monaco", "MC", "Principality of Monaco", "492",),
    "MDA": ("Moldova", "MD", "Moldova, Republic of", "Republic of Moldova", "498",),
    "MDG": ("Madagascar", "MG", "Republic of Madagascar", "450",),
    "MDV": ("Maldives", "MV", "Republic of Maldives", "462",),
    "MEX": ("Mexico", "MX", "United Mexican States", "484",),
    "MHL": ("Marshall Islands", "MH", "Republic of the Marshall Islands", "584",),
    "MKD": ("North Macedonia", "MK", "Republic of North Macedonia", "807",),
    "MLI": ("Mali", "ML", "Republic of Mali", "466",),
    "MLT": ("Malta", "MT", "Republic of Malta", "470",),
    "MMR": ("Myanmar", "MM", "Republic of Myanmar", "Republic of the Union of Myanmar", "104",),
    "MNE": ("Montenegro", "ME", "499",),
    "MNG": ("Mongolia", "MN", "496",),
    "MNP": ("Northern Mariana Islands", "MP", "Commonwealth of the Northern Mariana Islands", "580",),
    "MOZ": ("Mozambique", "MZ", "Republic of Mozambique", "508",),
    "MRT": ("Mauritania", "MR", "Islamic Republic of Mauritania", "478",),
    "MSR": ("Montserrat", "MS", "500",),
    "MTQ": ("Martinique", "MQ", "474",),
    "MUS": ("Mauritius", "MU", "Republic of Mauritius", "480",),
    "MWI": ("Malawi", "MW", "Republic of Malawi", "454",),
    "MYS": ("Malaysia", "MY", "458",),
    "MYT": ("Mayotte", "YT", "175",),
    "NAM": ("Namibia", "NA", "Republic of Namibia", "516",),
    "NCL": ("New Caledonia", "NC", "540",),
    "NER": ("Niger", "NE", "Republic of the Niger", "Republic of Niger", "562",),
    "NFK": ("Norfolk Island", "NF", "574",),
    "NGA": ("Nigeria", "NG", "Federal Republic of Nigeria", "566",),
    "NIC": ("Nicaragua", "NI", "Republic of Nicaragua", "558",),
    "NIU": ("Niue", "NU", "570",),
    "NLD": ("Netherlands", "NL", "Kingdom of the Netherlands", "Netherlands (Kingdom of the)", "528",),
    "NOR": ("Norway", "NO", "Kingdom of Norway", "578",),
    "NPL": ("Nepal", "NP", "Federal Democratic Republic of Nepal", "524",),
    "NRU": ("Nauru", "NR", "Republic of Nauru", "520",),
    "NZL": ("New Zealand", "NZ", "554",),
    "OMN": ("Oman", "OM", "Sultanate of Oman", "512",),
    "PAK": ("Pakistan", "PK", "Islamic Republic of Pakistan", "586",),
    "PAN": ("Panama", "PA", "Republic of Panama", "591",),
    "PCN": ("Pitcairn", "PN", "612",),
    "PER": ("Peru", "PE", "Republic of Peru", "604",),
    "PHL": ("Philippines", "PH", "Republic of the Philippines", "608",),
    "PLW": ("Palau", "PW", "Republic of Palau", "585",),
    "PNG": ("Papua New Guinea", "PG", "Independent State of Papua New Guinea", "598",),
    "POL": ("Poland", "PL", "Republic of Poland", "616",),
    "PRI": ("Puerto Rico", "PR", "630",),
    "PRK": ("North Korea", "KP", "Korea, Democratic People's Republic of", "Democratic People's Republic of Korea", "408",),
    "PRT": ("Portugal", "PT", "Portuguese Republic", "620",),
    "PRY": ("Paraguay", "PY", "Republic of Paraguay", "600",),
    "PSE": ("Palestine", "PS", "Palestine, State of", "the State of Palestine", "State of Palestine", "gaza", "gaza strip", "palestine state", "palestinian territories", "palestinian territory", "west bank", "west bank gaza", "275",),
    "PYF": ("French Polynesia", "PF", "258",),
    "QAT": ("Qatar", "QA", "State of Qatar", "634",),
    "REU": ("R\u00e9union", "RE", "638",),
    "ROU": ("Romania", "RO", "642",),
    "RUS": ("Russia", "RU", "Russian Federation", "643",),
    "RWA": ("Rwanda", "RW", "Rwandese Republic", "Republic of Rwanda", "646",),
    "SAU": ("Saudi Arabia", "SA", "Kingdom of Saudi Arabia", "682",),
    "SDN": ("Sudan", "SD", "Republic of the Sudan", "729",),
    "SEN": ("Senegal", "SN", "Republic of Senegal", "686",),
    "SGP": ("Singapore", "SG", "Republic of Singapore", "702",),
    "SGS": ("South Georgia and South Sandwich Is.", "GS", "South Georgia and the South Sandwich Islands", "239",),
    "SHN": ("St. Helena", "SH", "Saint Helena, Ascension and Tristan da Cunha", "Saint Helena", "654",),
    "SJM": ("Svalbard and Jan Mayen Islands", "SJ", "Svalbard and Jan Mayen", "744",),
    "SLB": ("Solomon Islands", "SB", "090",),
    "SLE": ("Sierra Leone", "SL", "Republic of Sierra Leone", "694",),
    "SLV": ("El Salvador", "SV", "Republic of El Salvador", "222",),
    "SMR": ("San Marino", "SM", "Republic of San Marino", "674",),
    "SOM": ("Somalia", "SO", "Federal Republic of Somalia", "706",),
    "SPM": ("St. Pierre and Miquelon", "PM", "Saint Pierre and Miquelon", "666",),
    "SRB": ("Serbia", "RS", "Republic of Serbia", "688",),
    "SSD": ("South Sudan", "SS", "Republic of South Sudan", "728",),
    "STP": ("Sao Tome and Principe", "ST", "Democratic Republic of Sao Tome and Principe", "678",),
    "SUR": ("Suriname", "SR", "Republic of Suriname", "740",),
    "SVK": ("Slovakia", "SK", "Slovak Republic", "703",),
    "SVN": ("Slovenia", "SI", "Republic of Slovenia", "705",),
    "SWE": ("Sweden", "SE", "Kingdom of Sweden", "752",),
    "SWZ": ("Eswatini", "SZ", "Kingdom of Eswatini", "748",),
    "SXM": ("Sint Maarten", "SX", "Sint Maarten (Dutch part)", "534",),
    "SYC": ("Seychelles", "SC", "Republic of Seychelles", "690",),
    "SYR": ("Syria", "SY", "Syrian Arab Republic", "760",),
    "TCA": ("Turks and Caicos Islands", "TC", "796",),
    "TCD": ("Chad", "TD", "Republic of Chad", "148",),
    "TGO": ("Togo", "TG", "Togolese Republic", "768",),
    "THA": ("Thailand", "TH", "Kingdom of Thailand", "764",),
    "TJK": ("Tajikistan", "TJ", "Republic of Tajikistan", "762",),
    "TKL": ("Tokelau", "TK", "772",),
    "TKM": ("Turkmenistan", "TM", "795",),
    "TLS": ("Timor-Leste", "TL", "Democratic Republic of Timor-Leste", "626",),
    "TON": ("Tonga", "TO", "Kingdom of Tonga", "776",),
    "TTO": ("Trinidad and Tobago", "TT", "Republic of Trinidad and Tobago", "780",),
    "TUN": ("Tunisia", "TN", "Republic of Tunisia", "788",),
    "TUR": ("Turkey", "TR", "T\u00fcrkiye", "Republic of T\u00fcrkiye", "792",),
    "TUV": ("Tuvalu", "TV", "798",),
    "TWN": ("Taiwan", "TW", "Taiwan, Province of China", "Republic of China", "158",),
    "TZA": ("Tanzania", "TZ", "Tanzania, United Republic of", "United Republic of Tanzania", "834",),
    "UGA": ("Uganda", "UG", "Republic of Uganda", "800",),
    "UKR": ("Ukraine", "UA", "804",),
    "UMI": ("United States Minor Outlying Islands", "UM", "581",),
    "URY": ("Uruguay", "UY", "Eastern Republic of Uruguay", "Oriental Republic of Uruguay", "858",),
    "USA": ("United States", "US", "United States of America", "usa", "u s", "u s a", "840",),
    "UZB": ("Uzbekistan", "UZ", "Republic of Uzbekistan", "860",),
    "VAT": ("Vatican", "VA", "Holy See (Vatican City State)", "Vatican City State", "Holy See", "336",),
    "VCT": ("St. Vincent and the Grenadines", "VC", "Saint Vincent and the Grenadines", "670",),
    "VEN": ("Venezuela", "VE", "Venezuela, Bolivarian Republic of", "Bolivarian Republic of Venezuela", "862",),
    "VGB": ("British Virgin Islands", "VG", "Virgin Islands, British", "092",),
    "VIR": ("United States Virgin Islands", "VI", "Virgin Islands, U.S.", "Virgin Islands of the United States", "850",),
    "VNM": ("Vietnam", "VN", "Viet Nam", "Socialist Republic of Viet Nam", "Socialist Republic of Vietnam", "704",),
    "VUT": ("Vanuatu", "VU", "Republic of Vanuatu", "548",),
    "WLF": ("Wallis and Futuna Islands", "WF", "Wallis and Futuna", "876",),
    "WSM": ("Samoa", "WS", "Independent State of Samoa", "882",),
    "XKX": ("Kosovo", "XK", "Republic of Kosovo",),
    "YEM": ("Yemen", "YE", "Republic of Yemen", "887",),
    "ZAF": ("South Africa", "ZA", "Republic of South Africa", "710",),
    "ZMB": ("Zambia", "ZM", "Republic of Zambia", "894",),
    "ZWE": ("Zimbabwe", "ZW", "Republic of Zimbabwe", "716",),
}


SYSTEM_PROMPT = """Classify and, only when appropriate, normalize one raw country-field value.
Determine whether the value itself names an entity in STANDARD_COUNTRIES_JSON; do not determine the country
that contains, governs, hosts or is associated with a place or organization.

Return only a JSON object with exactly these keys:
{"status":"ok|multi_country|non_country|unresolved","countries":[{"country_name":"short English name","iso3":"ISO-3166 alpha-3"}]}

- ok: the value unambiguously names one catalog country/territory, including
  a recognizable abbreviation, alternative spelling or name in another language.
- multi_country: a clear list of two or more distinct catalog entities. Resolve
  the whole list; never return a partially resolved mixed expression.
- non_country: the value names an entity outside the catalog, such as a city,
  province, region, continent, organization, sea, border area or historical state.
  A territory explicitly listed in the catalog remains an eligible entity.
- unresolved: the meaning is ambiguous or the catalog match cannot be determined.

For example, UAE can match United Arab Emirates, but Paris must not become
France, Karnataka must not become India, and European Union or United Nations
must not become member countries or headquarters countries. Korea without
disambiguating information is unresolved. Do not map a region to a parent or
claimant country or a historical state to a modern successor.

Never invent names or codes. Return JSON only.
STANDARD_COUNTRIES_JSON:
""" + json.dumps([
    {"country_name": COUNTRY_CATALOG[code][0], "iso3": code}
    for code in sorted(COUNTRY_CATALOG)
], ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass
class Settings:
    api_key: str | None = field(default=None, repr=False)
    llm: bool = False
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 1.0
    attempts: int = 3
    timeout: float = 180
    retry_delay: float = 1
    dry_run: bool = False


def mention_key(value):
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Preserve non-Latin names so distinct mentions cannot collapse to one cache key.
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def mention_identity(value):
    """Keep list punctuation meaningful when identifying cached mentions."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _build_country_index():
    index = {}
    for code, names in COUNTRY_CATALOG.items():
        for name in ((code, *names) if code == "XKX" else (code, names[0])):
            key = mention_key(name)
            if key in index and index[key] != code:
                raise ValueError(f"Ambiguous country catalog alias: {name!r}")
            index[key] = code
    return index


_COUNTRY_CODES = _build_country_index()


def _unique_object(pairs):
    if len(pairs) != len({key for key, _ in pairs}):
        raise ValueError("Duplicate JSON keys are not accepted.")
    return dict(pairs)


def _reject_constant(_):
    raise ValueError("Non-standard JSON numbers are not accepted.")


def strict_json(value):
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _result(status, countries=None, *, method="rules", error="", attempts=0):
    return {"status": status, "countries": countries or [], "method": method,
            "error": error, "attempts": attempts}


def _country(name, iso3):
    return {"country_name": name, "iso3": iso3}


def _lookup(raw, *, exact=True):
    """Resolve exact ISO names/codes first, then frozen preferred labels."""
    try:
        code = pycountry.countries.lookup(raw.strip()).alpha_3
    except LookupError:
        code = None
    if code in COUNTRY_CATALOG:
        return _country(COUNTRY_CATALOG[code][0], code)
    code = _COUNTRY_CODES.get(mention_key(raw))
    return _country(COUNTRY_CATALOG[code][0], code) if code else None


def deterministic_normalization(raw):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _result("blank")
    if not isinstance(raw, str):
        return _result("unresolved", error="Country mention must be a string or null.")
    country = _lookup(raw.strip(), exact=True)
    if country:
        return _result("ok", [country])
    return _result("unresolved", error="No exact country match; semantic classification is required.")


def parse_prediction(raw):
    obj = strict_json(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    if not isinstance(obj, dict) or set(obj) != {"status", "countries"}:
        raise ValueError("Expected exactly status and countries in the response.")
    status, countries = obj["status"], obj["countries"]
    if not isinstance(status, str) or status not in SUCCESS | {"unresolved"} or not isinstance(countries, list):
        raise ValueError("Invalid normalization status or country array.")
    if ((status == "ok" and len(countries) != 1)
            or (status == "multi_country" and len(countries) < 2)
            or (status in {"non_country", "unresolved"} and countries)):
        raise ValueError("Status and number of countries disagree.")
    seen = set()
    for country in countries:
        if (not isinstance(country, dict) or set(country) != {"country_name", "iso3"}
                or any(not isinstance(v, str) or not v.strip() for v in country.values())
                or not re.fullmatch(r"[A-Z]{3}", country["iso3"])):
            raise ValueError("Invalid country name/ISO3 pair.")
        resolved = _lookup(country["country_name"], exact=True)
        if resolved is None or resolved["iso3"] != country["iso3"] or country["iso3"] in seen:
            raise ValueError("Country name and ISO3 must match a distinct known country.")
        seen.add(country["iso3"])
        country.update(resolved)
    return obj


def validate_settings(settings):
    if (not isinstance(settings.attempts, int) or settings.attempts < 1
            or not math.isfinite(settings.timeout) or settings.timeout <= 0
            or not math.isfinite(settings.retry_delay) or settings.retry_delay < 0
            or not math.isfinite(settings.temperature) or not 0 <= settings.temperature <= 2):
        raise ValueError("Invalid attempts, timeout, retry-delay or temperature.")
    endpoint = urlsplit(settings.base_url)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username
            or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("base-url must be HTTP(S), without embedded credentials, query or fragment.")
    if not settings.model.strip():
        raise ValueError("Model cannot be empty.")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(settings):
    validate_settings(settings)
    return {"schema": "country_normalization_v3", "rules_sha256": _file_hash(__file__),
            "pycountry_version": pycountry.__version__,
            "prompt_sha256": json_hash(SYSTEM_PROMPT), "catalog_sha256": json_hash(COUNTRY_CATALOG),
            "model": settings.model,
            "temperature": settings.temperature, "base_url": settings.base_url.rstrip("/"),
            "llm": settings.llm}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_open(path):
    return (gzip.open if str(path).lower().endswith(".gz") else open)(
        path, "rt", encoding="utf-8-sig", newline="")


def _csv_reader(stream):
    # Article text can exceed the platform csv default (128 KiB).
    csv.field_size_limit(2 ** 31 - 1)
    reader = csv.DictReader(stream, strict=True)
    fields = reader.fieldnames
    if not fields or any(not f.strip() for f in fields) or len(fields) != len(set(fields)):
        raise ValueError("CSV requires unique, nonempty headers.")
    return reader


def _csv_rows(reader):
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row width does not match its header.")
        yield row


def _cached_mention(cache_dir, key, fingerprint):
    path = Path(cache_dir) / f"{json_hash(key)}.json"
    try:
        cached = strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint or cached.get("mention_key") != key:
            return None
        result = cached["result"]
        if (not isinstance(result, dict) or result.get("status") not in SUCCESS
                or result.get("method") not in {"rules", "llm"}
                or type(result.get("attempts")) is not int or result["attempts"] < 0
                or not isinstance(result.get("error"), str)
                or not isinstance(result.get("country_raw"), str)
                or mention_identity(result["country_raw"]) != key):
            return None
        validated = parse_prediction({"status": result["status"], "countries": result["countries"]})
        result["countries"] = validated["countries"]
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _legacy_mention_caches(roots, config):
    caches = []
    for root in dict.fromkeys(Path(root).expanduser().absolute() for root in roots):
        cache = root / "mention_cache"
        if cache.is_symlink() or not cache.is_dir():
            continue
        # Absent mentions may still carry a fingerprint from an earlier input.
        history = root / "input_revisions"
        manifests = [root / "manifest.json"]
        if not history.is_symlink():
            manifests.extend(sorted(history.glob("*/manifest.json")))
        for path in manifests:
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                manifest = strict_json(path.read_text(encoding="utf-8"))
                if (not isinstance(manifest, dict) or manifest.get("configuration") != config
                        or manifest.get("fingerprint") != json_hash({k: v for k, v in manifest.items() if k != "fingerprint"})):
                    continue
                caches.append((cache, json_hash(config)))
                caches.append((cache, manifest["fingerprint"]))
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return list(dict.fromkeys(caches))


class CountryNormalizer:
    def __init__(self, settings=None, *, request=None, cache_dir=None, fingerprint=None, legacy_caches=()):
        self.settings = settings or Settings()
        validate_settings(self.settings)
        self.request = request
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.fingerprint = fingerprint or json_hash(configuration(self.settings))
        self.legacy_caches = tuple(legacy_caches)
        self.results = {}
        self.occurrences = {}
        self.client = None
        self.requests = self.cache_reused = 0

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def _request(self, raw):
        content = json.dumps({"raw_country": raw}, ensure_ascii=False)
        if self.request is not None:
            return self.request(content)
        if self.client is None:
            self.client = _make_client(self.settings)
        response = self.client.chat.completions.create(
            model=self.settings.model, temperature=self.settings.temperature,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            response_format={"type": "json_object"}, stream=False)
        return response.choices[0].message.content

    def _save(self, key, result):
        if self.cache_dir is not None and not self.settings.dry_run:
            atomic_json(self.cache_dir / f"{json_hash(key)}.json", {
                "fingerprint": self.fingerprint, "mention_key": key, "result": result})

    def normalize(self, raw):
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return _result("blank")
        if not isinstance(raw, str):
            return deterministic_normalization(raw)
        key = mention_identity(raw) or raw
        self.occurrences[key] = self.occurrences.get(key, 0) + 1
        if key in self.results:
            return copy.deepcopy(self.results[key])
        result = deterministic_normalization(raw)
        if result["status"] == "unresolved" and self.settings.llm and not self.settings.dry_run:
            result = self._fallback(raw, key)
        result["country_raw"] = raw
        self.results[key] = result
        self._save(key, result)
        return copy.deepcopy(result)

    def _fallback(self, raw, key):
        if self.cache_dir is not None:
            result = _cached_mention(self.cache_dir, key, self.fingerprint)
            if result is not None:
                self.cache_reused += 1
                return result
            for cache, fingerprint in self.legacy_caches:
                if (cache / f"{json_hash(key)}.json").is_symlink():
                    continue
                result = _cached_mention(cache, key, fingerprint)
                if result is not None:
                    self.cache_reused += 1
                    return result
        result = _result("unresolved", method="llm")
        for attempt in range(1, self.settings.attempts + 1):
            self.requests += 1
            try:
                parsed = parse_prediction(self._request(raw))
                result = _result(parsed["status"], parsed["countries"], method="llm", attempts=attempt)
                if parsed["status"] == "unresolved":
                    result["error"] = "Model explicitly left the country unresolved."
            except Exception as error:
                result = _result("unresolved", method="llm", attempts=attempt,
                                 error=safe_error(error, self.settings.api_key))
            else:
                self._save(key, dict(result, country_raw=raw))
                break
            self._save(key, dict(result, country_raw=raw))
            if attempt < self.settings.attempts:
                time.sleep(self.settings.retry_delay * 2 ** (attempt - 1))
        return result


def normalize_mention(value, *, settings=None, request=None):
    normalizer = CountryNormalizer(settings, request=request)
    try:
        return normalizer.normalize(value)
    finally:
        normalizer.close()


def _decode(value, expected):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = strict_json(value) if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(parsed, expected):
        raise ValueError(f"Expected JSON {expected.__name__}.")
    return parsed


def _validate_entities(items, kind):
    if not isinstance(items, list):
        raise ValueError(f"{kind} must be a JSON array.")
    required = "name" if kind in {"state_actors", "individuals"} else None
    for item in items:
        if not isinstance(item, dict) or (required and required not in item):
            raise ValueError(f"{kind} entries must be objects with {required or 'location fields'}.")
        for attribute in ({"name", "country"} if required else {"country"}):
            if attribute in item and item[attribute] is not None and not isinstance(item[attribute], str):
                raise ValueError(f"{kind}.{attribute} must be a string or null.")


def _normalize_record(record, normalizer):
    out = copy.deepcopy(record)
    errors = []
    representations = {}

    def rewrite(items, kind, label):
        attribute = "name" if kind == "state_actors" else "country"
        for item in items:
            if attribute not in item:
                continue
            result = normalizer.normalize(item[attribute])
            if result["status"] in {"ok", "multi_country"}:
                item[attribute] = "; ".join(c["country_name"] for c in result["countries"])
                if kind not in {"state_actors", "individuals"}:
                    item["country_iso3"] = "; ".join(c["iso3"] for c in result["countries"])
            if result["status"] == "unresolved":
                errors.append(f"{label}: {item[attribute]!r}: {result['error']}")
        if kind in {"state_actors", "individuals"}:
            signature = sorted((str(item.get("name") or "").casefold(),
                                str(item.get("country") or "").casefold() if kind == "individuals" else "")
                               for item in items)
            if kind in representations and signature != representations[kind]:
                errors.append(f"Conflicting flattened and nested {kind} country representations.")
            representations[kind] = signature

    for label in (*LOCATION_FIELDS, "state_actors", "individuals", "inference", "entities"):
        if label not in out:
            continue
        original = out[label]
        try:
            nested = label in {"inference", "entities"}
            value = _decode(original, dict if nested else list)
            if value is None:
                continue
            if nested:
                if label == "inference" and "entities" not in value:
                    raise ValueError("inference.entities is missing.")
                entities = value["entities"] if label == "inference" else value
                if not isinstance(entities, dict):
                    raise ValueError("inference.entities must be an object.")
                for kind in ("state_actors", "individuals"):
                    if kind in entities:
                        _validate_entities(entities[kind], kind)
                for kind in ("state_actors", "individuals"):
                    if kind in entities:
                        rewrite(entities[kind], kind, f"{label}.{kind}")
            else:
                _validate_entities(value, label)
                rewrite(value, label, label)
            out[label] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) if isinstance(original, str) else value
        except (ValueError, TypeError) as error:
            errors.append(f"{label}: {safe_error(error, normalizer.settings.api_key)}")
            out[label] = original
    out[STATUS_FIELDS[0]] = "unresolved" if errors else "ok"
    out[STATUS_FIELDS[1]] = safe_error("; ".join(errors), normalizer.settings.api_key)
    return out


def normalize_records(records, settings=None, *, request=None, normalizer=None):
    """Return independent copies; JSON strings stay strings and native objects stay native."""
    owned = normalizer is None
    normalizer = normalizer or CountryNormalizer(settings, request=request)
    try:
        return [_normalize_record(record, normalizer) for record in records]
    finally:
        if owned:
            normalizer.close()


@contextlib.contextmanager
def atomic_csv(path, fields):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            yield writer
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser():
    parser = StepArgumentParser(step="6_normalize_countries", description=__doc__)
    parser.add_argument("--input", "--enriched-news", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true", help="Enable fallback for unresolved distinct mentions only.")
    parser.add_argument("--api-key", help="Runtime key; otherwise hidden input. Never read from env/YAML or saved.")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Prepare/inspect inputs without model requests or normalized article output.")
    return parser


def main(argv=None, *, request=None):
    args = build_parser().parse_args(argv)
    settings = Settings(api_key=args.api_key, llm=args.llm, model=args.model,
                        base_url=args.base_url.rstrip("/"), temperature=args.temperature,
                        attempts=args.attempts, timeout=args.timeout, retry_delay=args.retry_delay,
                        dry_run=args.dry_run)
    normalizer = None
    run_summary = None
    progress = None
    try:
        source, output = args.input.resolve(), args.output_dir.resolve()
        if (source == output or source.is_relative_to(output) or output.is_relative_to(source)
                or not source.is_file() or not source.name.lower().endswith((".csv", ".csv.gz"))):
            raise ValueError("Input must be CSV/CSV.GZ and separate from the output directory.")
        config = configuration(settings)
        shared_cache = getattr(args, "shared_cache_dir", None)
        cache_fingerprint = json_hash(config)
        cache_dir = Path(shared_cache).expanduser().resolve() / cache_fingerprint if shared_cache is not None else output / "mention_cache"
        if shared_cache is not None and (source == cache_dir or source.is_relative_to(cache_dir)
                                         or cache_dir.is_relative_to(source)):
            raise ValueError("Shared country cache must be separate from article input.")
        guard_step_selection(args)
        manifest = {"source": str(source), "input_sha256": _file_hash(source), "configuration": config}
        fingerprint = json_hash(manifest)
        manifest["fingerprint"] = fingerprint
        with _csv_open(source) as stream:
            reader = _csv_reader(stream)
            if not set(reader.fieldnames).intersection((*LOCATION_FIELDS, "state_actors", "inference", "entities")):
                raise ValueError("Input needs canonical locations, state_actors or nested inference/entities JSON.")
            fields = list(dict.fromkeys([*reader.fieldnames, *STATUS_FIELDS]))
            if output.exists() and not output.is_dir():
                raise ValueError("Output must be a directory.")
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                previous = strict_json(manifest_path.read_text(encoding="utf-8"))
                if (not isinstance(previous, dict) or previous.get("source") != str(source)
                        or previous.get("configuration") != config
                        or previous.get("fingerprint") != json_hash({k: v for k, v in previous.items()
                                                                     if k != "fingerprint"})):
                    raise ValueError("Input/configuration changed; use a new output directory.")
                if previous != manifest:
                    revision = output / "input_revisions" / json_hash(previous)
                    if not (revision / "manifest.json").exists():
                        atomic_json(revision / "manifest.json", previous)
                        summary_path = output / "run_summary.json"
                        if summary_path.exists():
                            atomic_json(revision / "run_summary.json",
                                        strict_json(summary_path.read_text(encoding="utf-8")))
            elif output.exists() and any(output.iterdir()):
                raise ValueError("Refusing output reuse without a matching manifest.")
            legacy_caches = _legacy_mention_caches(
                (output, *getattr(args, "legacy_cache_roots", ())), config)
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest_path, manifest)
            run_summary = {"status": "running", "records": 0, "unresolved_records": 0,
                           "fingerprint": fingerprint}
            atomic_json(output / "run_summary.json", run_summary)
            normalizer = CountryNormalizer(settings, request=request,
                                           cache_dir=cache_dir, fingerprint=cache_fingerprint,
                                           legacy_caches=legacy_caches)
            if not args.dry_run:
                with _csv_open(source) as counting_stream:
                    article_total = sum(1 for _ in _csv_rows(_csv_reader(counting_stream)))
                progress = tqdm(total=article_total, desc="Normalizing countries", unit="article",
                                ascii=True, dynamic_ncols=True, disable=False)
                progress.set_postfix(ai_requests=0, cache_reused=0)
            total = unresolved = 0
            article_output = (contextlib.nullcontext(None) if args.dry_run else
                              atomic_csv(output / "normalized_articles.csv", fields))
            with article_output as writer:
                for row in _csv_rows(reader):
                    result = _normalize_record(row, normalizer)
                    total += 1
                    unresolved += result[STATUS_FIELDS[0]] != "ok"
                    run_summary.update(records=total, unresolved_records=unresolved)
                    if writer is not None:
                        writer.writerow(result)
                    if progress is not None:
                        progress.set_postfix(ai_requests=normalizer.requests,
                                             cache_reused=normalizer.cache_reused, refresh=False)
                        progress.update(1)
                if _file_hash(source) != manifest["input_sha256"] or configuration(settings) != config:
                    raise ValueError("Input or configuration changed during normalization; output is not complete.")
        unresolved_fields = ["country_raw", "status", "method", "error", "attempts", "mention_count"]
        with atomic_csv(output / "unresolved_countries.csv", unresolved_fields) as writer:
            for key, result in normalizer.results.items():
                if result["status"] == "unresolved":
                    writer.writerow({**{f: result[f] for f in unresolved_fields if f != "mention_count"},
                                     "mention_count": normalizer.occurrences[key]})
        counts = {status: sum(r["status"] == status for r in normalizer.results.values())
                  for status in ("ok", "multi_country", "non_country", "unresolved")}
        summary = {"status": "prepared" if args.dry_run else "incomplete" if unresolved else "complete",
                   "stage": "dry-run" if args.dry_run else "normalize", "records": total,
                   "unresolved_records": unresolved, "distinct_mentions": len(normalizer.results),
                   "mention_status_counts": counts, "requests": normalizer.requests,
                   "cache_reused": normalizer.cache_reused, "fingerprint": fingerprint}
        atomic_json(output / "run_summary.json", summary)
        if progress is not None:
            progress.close()
        print(json.dumps(summary, indent=2))
        return 2 if unresolved and not args.dry_run else 0
    except Exception as error:
        if progress is not None:
            progress.close()
        message = safe_error(error, settings.api_key)
        if run_summary is not None:
            run_summary.update(status="failed", error=message)
            with contextlib.suppress(OSError):
                atomic_json(output / "run_summary.json", run_summary)
        print(f"Error: {message}", file=sys.stderr)
        return 1
    finally:
        try:
            if normalizer is not None:
                normalizer.close()
        finally:
            if progress is not None:
                progress.close()


if __name__ == "__main__":
    raise SystemExit(main())
