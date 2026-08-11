import json

import nvmolkit
import torch
from nvmolkit.fingerprints import MorganFingerprintGenerator
from rdkit import Chem


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

molecules = [
    Chem.MolFromSmiles(smiles)
    for smiles in ("CCO", "c1ccccc1", "CC(=O)O")
]
if any(molecule is None for molecule in molecules):
    raise RuntimeError("RDKit failed to parse a probe molecule")

fingerprint_generator = MorganFingerprintGenerator(radius=2, fpSize=1024)
fingerprint_result = fingerprint_generator.GetFingerprints(molecules)
torch.cuda.synchronize()
fingerprints = fingerprint_result.torch()

if tuple(fingerprints.shape) != (3, 32):
    raise RuntimeError(f"Unexpected fingerprint shape: {tuple(fingerprints.shape)}")

print(
    json.dumps(
        {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "dtype": str(fingerprints.dtype),
            "nvmolkit": nvmolkit.__version__,
            "shape": list(fingerprints.shape),
            "torch": torch.__version__,
        },
        sort_keys=True,
    )
)
