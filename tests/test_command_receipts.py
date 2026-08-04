from dataclasses import FrozenInstanceError

import pytest

from command_receipts import CommandReceipt, command_receipt
from demo_agent import (
    ClusterArgs,
    EmbedArgs,
    FingerprintArgs,
    InspectionArgs,
    OptimizationArgs,
    SimilarityArgs,
)


SECRET_BASIS = "Prefer this bounded choice nvapi-secret-that-must-not-render."


CASES = (
    (
        "inspect_library",
        InspectionArgs(),
        "inspect_library()",
        "RDKit inspection executed by Python",
        "inspect_library(state, DATA_PATH)",
    ),
    (
        "generate_morgan_fingerprints",
        FingerprintArgs(radius=2, size=1024, decision_basis=SECRET_BASIS),
        "generate_morgan_fingerprints(radius=2, size=1024)",
        "nvMolKit invocation executed by Python",
        (
            "generator = MorganFingerprintGenerator(radius=2, fpSize=1024)\n"
            "fingerprints = generator.GetFingerprints(molecules)"
        ),
    ),
    (
        "measure_tanimoto_similarity",
        SimilarityArgs(),
        "measure_tanimoto_similarity()",
        "nvMolKit invocation executed by Python",
        "similarity = crossTanimotoSimilarity(fingerprints)",
    ),
    (
        "discover_fused_butina_clusters",
        ClusterArgs(cutoff=0.5, decision_basis=SECRET_BASIS),
        "discover_fused_butina_clusters(cutoff=0.5)",
        "nvMolKit invocation executed by Python",
        "clusters = fused_butina(fingerprints.torch(), cutoff=0.5)[0]",
    ),
    (
        "embed_representative_conformers",
        EmbedArgs(
            representative_count=4,
            policy="include_singleton_if_available",
            conformers_per_representative=4,
            decision_basis=SECRET_BASIS,
        ),
        (
            "embed_representative_conformers(representative_count=4, "
            "policy='include_singleton_if_available', "
            "conformers_per_representative=4)"
        ),
        "nvMolKit invocation executed by Python",
        (
            "# Python/RDKit representative selection: count=4, "
            "policy='include_singleton_if_available'\n"
            "EmbedMolecules(molecules, parameters, confsPerMolecule=4, "
            "maxIterations=-1)"
        ),
    ),
    (
        "optimize_conformers_mmff94",
        OptimizationArgs(),
        "optimize_conformers_mmff94()",
        "nvMolKit invocation executed by Python",
        (
            "MMFFOptimizeMoleculesConfs(molecules, maxIters=500, "
            "output=CoordinateOutput.DEVICE)"
        ),
    ),
)


@pytest.mark.parametrize(
    ("stage", "arguments", "approved", "label", "invocation"), CASES
)
def test_command_receipts_match_all_exact_templates(
    stage, arguments, approved, label, invocation
):
    assert command_receipt(stage, arguments) == CommandReceipt(
        approved_tool_call=approved,
        scientific_label=label,
        scientific_invocation=invocation,
    )


@pytest.mark.parametrize(("stage", "arguments", "_approved", "_label", "_invocation"), CASES)
def test_command_receipts_exclude_model_text_secrets_and_runtime_reprs(
    stage, arguments, _approved, _label, _invocation
):
    rendered = repr(command_receipt(stage, arguments))

    assert "decision_basis" not in rendered
    assert "nvapi-" not in rendered
    assert "object at 0x" not in rendered
    assert type(arguments).__name__ not in rendered


@pytest.mark.parametrize(
    ("stage", "wrong_arguments"),
    [
        ("inspect_library", SimilarityArgs()),
        ("generate_morgan_fingerprints", ClusterArgs(cutoff=0.5, decision_basis="brief")),
        ("measure_tanimoto_similarity", OptimizationArgs()),
        ("discover_fused_butina_clusters", FingerprintArgs(radius=2, size=1024, decision_basis="brief")),
        (
            "embed_representative_conformers",
            FingerprintArgs(radius=2, size=1024, decision_basis="brief"),
        ),
        ("optimize_conformers_mmff94", InspectionArgs()),
    ],
)
def test_command_receipt_rejects_the_wrong_model_for_each_stage(
    stage, wrong_arguments
):
    with pytest.raises(ValueError, match="Arguments do not match workflow stage"):
        command_receipt(stage, wrong_arguments)


def test_command_receipt_rejects_unknown_stage_before_reading_arguments():
    class ExplosiveArguments:
        def __repr__(self):
            raise AssertionError("unsupported-stage arguments must not be rendered")

    with pytest.raises(ValueError, match=r"^Unsupported workflow stage\.$"):
        command_receipt("invented_stage", ExplosiveArguments())


@pytest.mark.parametrize(("stage", "arguments", "_approved", "_label", "_invocation"), CASES)
def test_command_receipt_is_deterministic(stage, arguments, _approved, _label, _invocation):
    assert command_receipt(stage, arguments) == command_receipt(stage, arguments)


def test_command_receipt_is_frozen():
    receipt = command_receipt("inspect_library", InspectionArgs())

    with pytest.raises(FrozenInstanceError):
        receipt.approved_tool_call = "changed"
