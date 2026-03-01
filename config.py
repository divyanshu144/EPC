from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parent

# Raw EPC files (zipped)
RAW_BULK_DIR = ROOT / "data" / "raw" / "ew_bulk"

# Extracted EPC folders (unzipped)
RAW_EXTRACT_DIR = ROOT / "data" / "raw" / "ew_extracted"

# Merged full EPC core CSV
EPC_MERGED_CSV = ROOT / "data" / "processed" / "ew_epc_core.csv"

# Cleaned South West outputs directory (CSV and derived files)
EPC_CLEAN_SW_DIR = ROOT / "data" / "processed" / "ew_epc_core_clean_sw"

# Artifacts
ARTIFACTS_DIR = ROOT / "reports" / "artifacts"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.jsonl"

# South West LA list
SOUTH_WEST_LAS = [
    "Bath and North East Somerset",
    "Bournemouth, Christchurch and Poole",
    "Bristol, City of",
    "Cornwall", "Devon", "Dorset", "Gloucestershire",
    "North Somerset", "Plymouth", "Somerset",
    "South Gloucestershire", "Swindon", "Torbay", "Wiltshire",
]

# Rating conversion
BAND_TO_SCORE = {"A":7,"B":6,"C":5,"D":4,"E":3,"F":2,"G":1}

# Normalised fuel and tenure maps
FUEL_MAP = {
    "mains gas": "gas",
    "mains gas (not community)": "gas",
    "mains gas - this is for backwards compatibility only": "gas",
    "electricity": "electricity",
    "LPG": "lpg",
    "oil": "oil",
    "biomass": "biomass",
    "community heat": "community_heat",
}

TENURE_MAP = {
    "owner-occupied": "Owner-occupied",
    "Owner-occupied": "Owner-occupied",
    "rental (private)": "Rented (private)",
    "rental": "Rented (private)",
    "rented (private)": "Rented (private)",
    "rented (social)": "Rented (social)",
    "social housing": "Rented (social)",
    "unknown": "Unknown",
    "Unknown": "Unknown",
}
