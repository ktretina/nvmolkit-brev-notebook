"""Shared ReFRAME loading and descriptor helpers for the ACS nvMolKit workshop."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


REFRAME_URL = "https://reframedb.org/assets/csv/reframe_smiles_list.csv"
SNAPSHOT_PATH = Path(__file__).with_name("data") / "reframe_teaching_snapshot.csv"
REQUIRED_COLUMNS = {
    "smile",
    "canonical_ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
}


def load_reframe(sample_size=2048, anchor_terms=(), random_state=2026):
    """Load, deduplicate, deterministically sample, and parse ReFRAME compounds."""
    local_csv = os.environ.get("REFRAME_CSV", "")
    if local_csv:
        try:
            frame = pd.read_csv(local_csv)
        except Exception:
            raise ValueError("Could not load the configured ReFRAME CSV.") from None
        source = str(Path(local_csv).resolve())
    else:
        try:
            frame = pd.read_csv(REFRAME_URL)
            source = REFRAME_URL
        except Exception:
            warnings.warn(
                "Could not load the live ReFRAME export; "
                "using the shared teaching snapshot."
            )
            frame = pd.read_csv(SNAPSHOT_PATH)
            source = f"shared {len(frame)}-row teaching snapshot"

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"ReFRAME schema changed; missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["name"] = frame["name"].fillna("unnamed compound")
    frame = frame.dropna(subset=["smile", "canonical_ikey"])
    frame = frame.drop_duplicates("canonical_ikey", keep="first").reset_index(drop=True)

    anchors = []
    for term in anchor_terms:
        hit = frame[frame["name"].str.contains(term, case=False, regex=False)]
        if not hit.empty:
            anchors.append(hit.iloc[[0]])
    anchor_frame = pd.concat(anchors, ignore_index=True) if anchors else frame.iloc[0:0]

    target_n = min(int(sample_size), len(frame))
    remaining = frame[~frame["canonical_ikey"].isin(anchor_frame["canonical_ikey"])]
    random_n = max(0, target_n - len(anchor_frame))
    sampled = remaining.sample(
        n=min(random_n, len(remaining)), random_state=random_state
    )
    frame = pd.concat([anchor_frame, sampled], ignore_index=True).drop_duplicates(
        "canonical_ikey"
    )

    molecules = [Chem.MolFromSmiles(smiles) for smiles in frame["smile"]]
    valid = np.asarray([molecule is not None for molecule in molecules])
    frame = frame.loc[valid].reset_index(drop=True)
    frame["_mol"] = [molecule for molecule in molecules if molecule is not None]
    frame.attrs.update(source=source, invalid_count=int((~valid).sum()))
    return frame


def add_descriptors(frame):
    """Add a compact set of interpretable physicochemical descriptors."""
    result = frame.copy()
    molecules = result["_mol"].tolist()
    result["MolWt"] = [Descriptors.MolWt(molecule) for molecule in molecules]
    result["cLogP"] = [Crippen.MolLogP(molecule) for molecule in molecules]
    result["TPSA"] = [rdMolDescriptors.CalcTPSA(molecule) for molecule in molecules]
    result["HBD"] = [Lipinski.NumHDonors(molecule) for molecule in molecules]
    result["HBA"] = [Lipinski.NumHAcceptors(molecule) for molecule in molecules]
    result["RotB"] = [Lipinski.NumRotatableBonds(molecule) for molecule in molecules]
    return result
