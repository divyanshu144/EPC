"""Map the national full-load archive's schema onto the pipeline's vocabulary.

The full-load archive replaced the per-authority and monthly archives, and it
does not use the same column names. Verified against the real 2020 and 2025
files (93 columns each):

* every name is lowercase; the monthly archives were uppercase
* ``LMK_KEY``                   -> ``certificate_number``
* ``BUILDING_REFERENCE_NUMBER`` -> ``uprn``
* ``COUNTY``                    -> absent (``region``/``country`` exist, but they
  are different things and nothing downstream reads ``COUNTY``)
* ``LOW_ENERGY_FIXED_LIGHT_COUNT`` -> ``low_energy_fixed_lighting_outlets_count``

Mapping happens here, at the ingestion boundary, so the rest of the pipeline
keeps working in one vocabulary rather than learning a second.

**The identifier change is the consequential one.**
``BUILDING_REFERENCE_NUMBER`` is the key behind the leakage-safe train/test
split in ``train_models.sample_data``: the same dwelling is certified repeatedly
over the years, and splitting by row rather than by property would score a model
partly on dwellings it had memorised.

``uprn`` was chosen on evidence, not resemblance:

* 1.7% null across a 500,000-row sample
* 4,037 UPRNs repeat within a single year, so it identifies a dwelling rather
  than a certificate
* 1,884 UPRNs appear in both 2012 and 2020, so it links the same dwelling across
  years — the property the split actually depends on

Two caveats worth carrying: 30% of UPRNs are ``Address Matched`` rather than
supplied by the assessor, so linkage is imperfect and some same-dwelling pairs
will be missed; and the cross-year overlap above was measured on the head of
each file, so it is a lower bound rather than the true rate.
"""

from __future__ import annotations

import pandas as pd

# Columns whose name changed outright. Case-only differences are handled
# generically and deliberately not listed here.
FULL_LOAD_RENAMES: dict[str, str] = {
    "LMK_KEY": "certificate_number",
    "BUILDING_REFERENCE_NUMBER": "uprn",
}

# Marks a row whose dwelling could not be identified. Prefixed rather than left
# blank so the two populations stay distinguishable downstream.
SYNTHETIC_KEY_PREFIX = "NOUPRN:"


def resolve_usecols(wanted: list[str], header: list[str]) -> tuple[list[str], dict[str, str]]:
    """Translate wanted canonical columns into names present in ``header``.

    Returns the source-side names to read, and a mapping back to canonical
    names. A wanted column with no counterpart is omitted rather than
    substituted — ``COUNTY`` has no equivalent, and quietly reading ``region``
    in its place would put different data under a familiar name.
    """
    available = {name.lower(): name for name in header}

    usecols: list[str] = []
    rename_map: dict[str, str] = {}
    for canonical in wanted:
        source = FULL_LOAD_RENAMES.get(canonical, canonical).lower()
        actual = available.get(source)
        if actual is None:
            continue
        usecols.append(actual)
        rename_map[actual] = canonical

    return usecols, rename_map


def normalise_full_load_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical column names and a usable property key.

    Idempotent: a frame already in the canonical vocabulary — as the monthly
    archives are — passes through untouched, so both archive shapes can coexist
    during a transition.
    """
    reverse = {source: canonical for canonical, source in FULL_LOAD_RENAMES.items()}

    renamed = {}
    for column in df.columns:
        if column in reverse:
            renamed[column] = reverse[column]
        elif column.isupper():
            renamed[column] = column
        else:
            renamed[column] = column.upper()

    out = df.rename(columns=renamed)

    if "BUILDING_REFERENCE_NUMBER" in out.columns:
        out["BUILDING_REFERENCE_NUMBER"] = _fill_missing_property_keys(
            out["BUILDING_REFERENCE_NUMBER"],
            out["LMK_KEY"] if "LMK_KEY" in out.columns else None,
        )

    return out


def _fill_missing_property_keys(keys: pd.Series, certificate_ids: pd.Series | None):
    """Give every unidentified dwelling its own key.

    1.7% of rows carry no UPRN. Leaving them blank would make the splitter treat
    them as a single enormous property, which creates the exact leakage the
    property-level split exists to prevent — worse than dropping them. Each gets
    a distinct synthetic key instead, so the rows stay usable and can never
    co-occur across the split.
    """
    keys = keys.astype("string")
    blank = keys.isna() | (keys.str.strip() == "")
    if not blank.any():
        return keys

    if certificate_ids is not None:
        fallback = SYNTHETIC_KEY_PREFIX + certificate_ids.astype("string")
    else:
        fallback = pd.Series(
            [f"{SYNTHETIC_KEY_PREFIX}row{i}" for i in range(len(keys))],
            index=keys.index,
            dtype="string",
        )

    return keys.where(~blank, fallback)
