import pytest

from demo_agent import FingerprintArgs, InspectionArgs, StageProposal


def test_controls_are_bounded_and_initialized():
    from interactive_workflow import controls_for

    assert controls_for(StageProposal("inspect_library", InspectionArgs())) == {}
    controls = controls_for(StageProposal(
        "generate_morgan_fingerprints",
        FingerprintArgs(radius=3, size=2048, decision_basis="validated"),
    ))
    assert tuple(controls) == ("radius", "size")
    assert controls["radius"].options == (2, 3)
    assert controls["radius"].value == 3
    assert controls["size"].options == (1024, 2048)


def test_controls_reject_stage_model_mismatch():
    from interactive_workflow import controls_for

    with pytest.raises(ValueError, match="match"):
        controls_for(StageProposal("inspect_library", FingerprintArgs(
            radius=2, size=1024, decision_basis="validated"
        )))
