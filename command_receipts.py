"""Deterministic display receipts for validated scientific workflow calls."""

from dataclasses import dataclass

from demo_agent import (
    ClusterArgs,
    EmbedArgs,
    FingerprintArgs,
    InspectionArgs,
    OptimizationArgs,
    SimilarityArgs,
)


_NVMOLKIT_LABEL = "nvMolKit invocation executed by Python"


@dataclass(frozen=True)
class CommandReceipt:
    approved_tool_call: str
    scientific_label: str
    scientific_invocation: str


def _require_exact_model(arguments: object, expected: type[object]) -> None:
    if type(arguments) is not expected:
        raise ValueError("Arguments do not match workflow stage.")


def command_receipt(stage: str, arguments: object) -> CommandReceipt:
    """Render one receipt from an allow-listed stage and its validated model."""
    if stage == "inspect_library":
        _require_exact_model(arguments, InspectionArgs)
        return CommandReceipt(
            approved_tool_call="inspect_library()",
            scientific_label="RDKit inspection executed by Python",
            scientific_invocation="inspect_library(state, DATA_PATH)",
        )

    if stage == "generate_morgan_fingerprints":
        _require_exact_model(arguments, FingerprintArgs)
        radius = repr(arguments.radius)
        size = repr(arguments.size)
        return CommandReceipt(
            approved_tool_call=(
                f"generate_morgan_fingerprints(radius={radius}, size={size})"
            ),
            scientific_label=_NVMOLKIT_LABEL,
            scientific_invocation=(
                f"generator = MorganFingerprintGenerator(radius={radius}, "
                f"fpSize={size})\n"
                "fingerprints = generator.GetFingerprints(molecules)"
            ),
        )

    if stage == "measure_tanimoto_similarity":
        _require_exact_model(arguments, SimilarityArgs)
        return CommandReceipt(
            approved_tool_call="measure_tanimoto_similarity()",
            scientific_label=_NVMOLKIT_LABEL,
            scientific_invocation=(
                "similarity = crossTanimotoSimilarity(fingerprints)"
            ),
        )

    if stage == "discover_fused_butina_clusters":
        _require_exact_model(arguments, ClusterArgs)
        cutoff = repr(arguments.cutoff)
        return CommandReceipt(
            approved_tool_call=(
                f"discover_fused_butina_clusters(cutoff={cutoff})"
            ),
            scientific_label=_NVMOLKIT_LABEL,
            scientific_invocation=(
                "clusters = fused_butina(fingerprints.torch(), "
                f"cutoff={cutoff})[0]"
            ),
        )

    if stage == "embed_representative_conformers":
        _require_exact_model(arguments, EmbedArgs)
        count = repr(arguments.representative_count)
        policy = repr(arguments.policy)
        conformer_count = repr(arguments.conformers_per_representative)
        return CommandReceipt(
            approved_tool_call=(
                "embed_representative_conformers("
                f"representative_count={count}, policy={policy}, "
                f"conformers_per_representative={conformer_count})"
            ),
            scientific_label=_NVMOLKIT_LABEL,
            scientific_invocation=(
                "# Python/RDKit representative selection: "
                f"count={count}, policy={policy}\n"
                "EmbedMolecules(molecules, parameters, "
                f"confsPerMolecule={conformer_count}, maxIterations=-1)"
            ),
        )

    if stage == "optimize_conformers_mmff94":
        _require_exact_model(arguments, OptimizationArgs)
        return CommandReceipt(
            approved_tool_call="optimize_conformers_mmff94()",
            scientific_label=_NVMOLKIT_LABEL,
            scientific_invocation=(
                "MMFFOptimizeMoleculesConfs(molecules, maxIters=500, "
                "output=CoordinateOutput.DEVICE)"
            ),
        )

    raise ValueError("Unsupported workflow stage.")
