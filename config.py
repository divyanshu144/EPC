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

# ---------------------------------------------------------------------------
# South West geography — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------
# EPC bulk data is published per lower-tier / unitary authority (ONS GSS codes
# E06xxxxxx = unitary, E07xxxxxx = lower-tier district). County councils
# (E10xxxxxx) do NOT issue EPCs and have no bulk file, so they must never be used
# for download selection. Several of the dissertation's 14 South West analysis
# authorities are two-tier counties (Devon, Gloucestershire) or were created by
# reorganisation (Dorset 2019, BCP 2019, Somerset 2023); their EPC data lives in
# the constituent district files and must be rolled up.
#
# This mapping is used consistently by BOTH:
#   - scripts/pipeline/epc_download_merge_fast.py  (select bulk files by GSS code)
#   - scripts/pipeline/clean_epc.py                (filter rows + roll up labels)
#
# Each entry: GSS code -> {download_name, epc_label, analysis_authority}.
#   download_name      : human name of the EPC download unit (for messages).
#   epc_label          : value expected in the certificate LOCAL_AUTHORITY_LABEL
#                        (used as a fallback match when the GSS code is absent).
#   analysis_authority : one of the 14 dissertation clusters this rolls up to.
#
# NOTE: the E07 district GSS codes follow the standard ONS register but could
# NOT be validated against the live EPC file listing in this environment — they
# require live EPC/API confirmation (see report / remaining checks).
SOUTH_WEST_EPC_AUTHORITIES: dict[str, dict[str, str]] = {
    # --- Unitary authorities (1:1 with their analysis authority) ---
    "E06000022": {
        "download_name": "Bath and North East Somerset",
        "epc_label": "Bath and North East Somerset",
        "analysis_authority": "Bath and North East Somerset",
    },
    "E06000023": {
        "download_name": "Bristol, City of",
        "epc_label": "Bristol, City of",
        "analysis_authority": "Bristol, City of",
    },
    "E06000024": {
        "download_name": "North Somerset",
        "epc_label": "North Somerset",
        "analysis_authority": "North Somerset",
    },
    "E06000025": {
        "download_name": "South Gloucestershire",
        "epc_label": "South Gloucestershire",
        "analysis_authority": "South Gloucestershire",
    },
    "E06000026": {
        "download_name": "Plymouth",
        "epc_label": "Plymouth",
        "analysis_authority": "Plymouth",
    },
    "E06000027": {"download_name": "Torbay", "epc_label": "Torbay", "analysis_authority": "Torbay"},
    "E06000030": {
        "download_name": "Swindon",
        "epc_label": "Swindon",
        "analysis_authority": "Swindon",
    },
    "E06000052": {
        "download_name": "Cornwall",
        "epc_label": "Cornwall",
        "analysis_authority": "Cornwall",
    },
    "E06000054": {
        "download_name": "Wiltshire",
        "epc_label": "Wiltshire",
        "analysis_authority": "Wiltshire",
    },
    # --- Bournemouth, Christchurch and Poole (BCP) ---
    # Pre-2019: Bournemouth & Poole were unitary, Christchurch a Dorset district;
    # all roll up to BCP so the analysis cluster is stable across the panel.
    "E06000058": {
        "download_name": "Bournemouth, Christchurch and Poole",
        "epc_label": "Bournemouth, Christchurch and Poole",
        "analysis_authority": "Bournemouth, Christchurch and Poole",
    },
    "E06000028": {
        "download_name": "Bournemouth",
        "epc_label": "Bournemouth",
        "analysis_authority": "Bournemouth, Christchurch and Poole",
    },
    "E06000029": {
        "download_name": "Poole",
        "epc_label": "Poole",
        "analysis_authority": "Bournemouth, Christchurch and Poole",
    },
    "E07000048": {
        "download_name": "Christchurch",
        "epc_label": "Christchurch",
        "analysis_authority": "Bournemouth, Christchurch and Poole",
    },
    # --- Dorset (unitary 2019+, formerly Dorset county districts) ---
    "E06000059": {"download_name": "Dorset", "epc_label": "Dorset", "analysis_authority": "Dorset"},
    "E07000049": {
        "download_name": "East Dorset",
        "epc_label": "East Dorset",
        "analysis_authority": "Dorset",
    },
    "E07000050": {
        "download_name": "North Dorset",
        "epc_label": "North Dorset",
        "analysis_authority": "Dorset",
    },
    "E07000051": {
        "download_name": "Purbeck",
        "epc_label": "Purbeck",
        "analysis_authority": "Dorset",
    },
    "E07000052": {
        "download_name": "West Dorset",
        "epc_label": "West Dorset",
        "analysis_authority": "Dorset",
    },
    "E07000053": {
        "download_name": "Weymouth and Portland",
        "epc_label": "Weymouth and Portland",
        "analysis_authority": "Dorset",
    },
    # --- Devon (two-tier county -> 8 lower-tier districts) ---
    "E07000040": {
        "download_name": "East Devon",
        "epc_label": "East Devon",
        "analysis_authority": "Devon",
    },
    "E07000041": {"download_name": "Exeter", "epc_label": "Exeter", "analysis_authority": "Devon"},
    "E07000042": {
        "download_name": "Mid Devon",
        "epc_label": "Mid Devon",
        "analysis_authority": "Devon",
    },
    "E07000043": {
        "download_name": "North Devon",
        "epc_label": "North Devon",
        "analysis_authority": "Devon",
    },
    "E07000044": {
        "download_name": "South Hams",
        "epc_label": "South Hams",
        "analysis_authority": "Devon",
    },
    "E07000045": {
        "download_name": "Teignbridge",
        "epc_label": "Teignbridge",
        "analysis_authority": "Devon",
    },
    "E07000046": {
        "download_name": "Torridge",
        "epc_label": "Torridge",
        "analysis_authority": "Devon",
    },
    "E07000047": {
        "download_name": "West Devon",
        "epc_label": "West Devon",
        "analysis_authority": "Devon",
    },
    # --- Gloucestershire (two-tier county -> 6 lower-tier districts) ---
    "E07000078": {
        "download_name": "Cheltenham",
        "epc_label": "Cheltenham",
        "analysis_authority": "Gloucestershire",
    },
    "E07000079": {
        "download_name": "Cotswold",
        "epc_label": "Cotswold",
        "analysis_authority": "Gloucestershire",
    },
    "E07000080": {
        "download_name": "Forest of Dean",
        "epc_label": "Forest of Dean",
        "analysis_authority": "Gloucestershire",
    },
    "E07000081": {
        "download_name": "Gloucester",
        "epc_label": "Gloucester",
        "analysis_authority": "Gloucestershire",
    },
    "E07000082": {
        "download_name": "Stroud",
        "epc_label": "Stroud",
        "analysis_authority": "Gloucestershire",
    },
    "E07000083": {
        "download_name": "Tewkesbury",
        "epc_label": "Tewkesbury",
        "analysis_authority": "Gloucestershire",
    },
    # --- Somerset (unitary 2023+, formerly Somerset county districts) ---
    # Somerset West and Taunton (E07000246) was itself formed in 2019 from the
    # merger of Taunton Deane (E07000190) and West Somerset (E07000191); all
    # three historical codes are included to cover the full 2008–2025 panel.
    "E06000066": {
        "download_name": "Somerset",
        "epc_label": "Somerset",
        "analysis_authority": "Somerset",
    },
    "E07000187": {
        "download_name": "Mendip",
        "epc_label": "Mendip",
        "analysis_authority": "Somerset",
    },
    "E07000188": {
        "download_name": "Sedgemoor",
        "epc_label": "Sedgemoor",
        "analysis_authority": "Somerset",
    },
    "E07000189": {
        "download_name": "South Somerset",
        "epc_label": "South Somerset",
        "analysis_authority": "Somerset",
    },
    "E07000246": {
        "download_name": "Somerset West and Taunton",
        "epc_label": "Somerset West and Taunton",
        "analysis_authority": "Somerset",
    },
    "E07000190": {
        "download_name": "Taunton Deane",
        "epc_label": "Taunton Deane",
        "analysis_authority": "Somerset",
    },
    "E07000191": {
        "download_name": "West Somerset",
        "epc_label": "West Somerset",
        "analysis_authority": "Somerset",
    },
}

# The 14 dissertation analysis authorities (== FE clusters), derived from the map.
ANALYSIS_AUTHORITIES: list[str] = sorted(
    {v["analysis_authority"] for v in SOUTH_WEST_EPC_AUTHORITIES.values()}
)

# Invariant: the mapping must roll up to exactly the 14 expected authorities.
_EXPECTED_ANALYSIS_AUTHORITIES = {
    "Bath and North East Somerset",
    "Bournemouth, Christchurch and Poole",
    "Bristol, City of",
    "Cornwall",
    "Devon",
    "Dorset",
    "Gloucestershire",
    "North Somerset",
    "Plymouth",
    "Somerset",
    "South Gloucestershire",
    "Swindon",
    "Torbay",
    "Wiltshire",
}
assert set(ANALYSIS_AUTHORITIES) == _EXPECTED_ANALYSIS_AUTHORITIES, (
    "SOUTH_WEST_EPC_AUTHORITIES must roll up to exactly the 14 dissertation "
    f"authorities. Got: {sorted(set(ANALYSIS_AUTHORITIES) ^ _EXPECTED_ANALYSIS_AUTHORITIES)}"
)

# Derived lookups (no pandas dependency — usable anywhere).
SOUTH_WEST_EPC_CODES: set[str] = set(SOUTH_WEST_EPC_AUTHORITIES)
CODE_TO_ANALYSIS_AUTHORITY: dict[str, str] = {
    code: meta["analysis_authority"] for code, meta in SOUTH_WEST_EPC_AUTHORITIES.items()
}
LABEL_TO_ANALYSIS_AUTHORITY: dict[str, str] = {
    meta["epc_label"]: meta["analysis_authority"] for meta in SOUTH_WEST_EPC_AUTHORITIES.values()
}


def analysis_authority_for(code: str | None = None, label: str | None = None) -> str | None:
    """Roll a raw EPC (code, label) up to one of the 14 analysis authorities.

    Matches on GSS code first (robust), then falls back to the certificate
    LOCAL_AUTHORITY_LABEL. Returns None if the row is not a South West authority.
    """
    if code is not None and code in CODE_TO_ANALYSIS_AUTHORITY:
        return CODE_TO_ANALYSIS_AUTHORITY[code]
    if label is not None and label in LABEL_TO_ANALYSIS_AUTHORITY:
        return LABEL_TO_ANALYSIS_AUTHORITY[label]
    return None


# Backward-compatible alias: SOUTH_WEST_LAS now means the 14 analysis-level
# authorities (post-rollup), which is what downstream code treats as the LA set.
SOUTH_WEST_LAS = ANALYSIS_AUTHORITIES

# Rating conversion
BAND_TO_SCORE = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}

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
