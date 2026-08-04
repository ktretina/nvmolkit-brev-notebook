import ast
from dataclasses import FrozenInstanceError
import inspect
import textwrap

import pytest

import chemistry_workflow
import demo_agent
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


def test_command_receipt_rejects_a_subclass_of_the_allowed_model():
    class InspectionArgsSubclass(InspectionArgs):
        pass

    with pytest.raises(ValueError, match="Arguments do not match workflow stage"):
        command_receipt("inspect_library", InspectionArgsSubclass())


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


@pytest.mark.parametrize(
    ("arguments", "approved", "invocation"),
    [
        (
            FingerprintArgs(radius=3, size=2048, decision_basis="alternate"),
            "generate_morgan_fingerprints(radius=3, size=2048)",
            "generator = MorganFingerprintGenerator(radius=3, fpSize=2048)\n"
            "fingerprints = generator.GetFingerprints(molecules)",
        ),
    ],
)
def test_fingerprint_receipt_formats_alternate_allowed_values(
    arguments, approved, invocation
):
    receipt = command_receipt("generate_morgan_fingerprints", arguments)

    assert receipt.approved_tool_call == approved
    assert receipt.scientific_invocation == invocation
    assert receipt == command_receipt("generate_morgan_fingerprints", arguments)


@pytest.mark.parametrize("cutoff", [0.4, 0.6])
def test_cluster_receipt_formats_both_allowed_boundaries(cutoff):
    arguments = ClusterArgs(cutoff=cutoff, decision_basis="boundary")
    receipt = command_receipt("discover_fused_butina_clusters", arguments)

    assert receipt.approved_tool_call == (
        f"discover_fused_butina_clusters(cutoff={cutoff!r})"
    )
    assert receipt.scientific_invocation == (
        f"clusters = fused_butina(fingerprints.torch(), cutoff={cutoff!r})[0]"
    )
    assert receipt == command_receipt("discover_fused_butina_clusters", arguments)


@pytest.mark.parametrize(
    ("count", "policy", "conformers"),
    [
        (3, "largest_clusters_first", 3),
        (6, "include_singleton_if_available", 8),
    ],
)
def test_embed_receipt_formats_boundaries_and_both_policies(
    count, policy, conformers
):
    arguments = EmbedArgs(
        representative_count=count,
        policy=policy,
        conformers_per_representative=conformers,
        decision_basis="boundary",
    )
    receipt = command_receipt("embed_representative_conformers", arguments)

    assert receipt.approved_tool_call == (
        "embed_representative_conformers("
        f"representative_count={count!r}, policy={policy!r}, "
        f"conformers_per_representative={conformers!r})"
    )
    assert receipt.scientific_invocation == (
        "# Python/RDKit representative selection: "
        f"count={count!r}, policy={policy!r}\n"
        "EmbedMolecules(molecules, parameters, "
        f"confsPerMolecule={conformers!r}, maxIterations=-1)"
    )
    assert receipt == command_receipt("embed_representative_conformers", arguments)


def _source_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _call_name(call):
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Call):
        return _call_name(target)
    raise AssertionError(f"Unsupported call target: {ast.dump(target)}")


def _one_source_call(function, name, *, keyword_names=None):
    matches = [
        node
        for node in ast.walk(_source_tree(function))
        if isinstance(node, ast.Call) and _call_name(node) == name
        and (
            keyword_names is None
            or {keyword.arg for keyword in node.keywords} == keyword_names
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _one_receipt_call(receipt, name):
    matches = [
        node
        for node in ast.walk(ast.parse(receipt.scientific_invocation))
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]
    assert len(matches) == 1
    return matches[0]


def _approved_call(receipt):
    statement = ast.parse(receipt.approved_tool_call).body[0]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    return statement.value


def _keyword_sources(call):
    return {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}


def _literal_keywords(call):
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}


def _import_contract(function):
    imports = [
        node for node in ast.walk(_source_tree(function))
        if isinstance(node, ast.ImportFrom)
    ]
    assert len(imports) == 1
    return imports[0].module, [alias.name for alias in imports[0].names]


def _return_expression(function):
    returns = [
        node.value for node in ast.walk(_source_tree(function))
        if isinstance(node, ast.Return)
    ]
    assert len(returns) == 1
    return returns[0]


def _assigned_expression(function, target_name):
    matches = [
        node.value
        for node in ast.walk(_source_tree(function))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _controller_mapping(stage):
    tree = _source_tree(demo_agent._executor_arguments)
    for branch in (node for node in ast.walk(tree) if isinstance(node, ast.If)):
        comparison = branch.test
        if (
            isinstance(comparison, ast.Compare)
            and len(comparison.comparators) == 1
            and isinstance(comparison.comparators[0], ast.Constant)
            and comparison.comparators[0].value == stage
        ):
            returned = next(
                node.value for node in branch.body if isinstance(node, ast.Return)
            )
            assert isinstance(returned, ast.Dict)
            return {
                ast.literal_eval(key): ast.unparse(value)
                for key, value in zip(returned.keys, returned.values)
            }
    raise AssertionError(f"No controller mapping found for {stage}")


@pytest.mark.parametrize(
    ("stage", "arguments", "field_to_workflow_keyword"),
    [
        (
            "generate_morgan_fingerprints",
            FingerprintArgs(radius=3, size=2048, decision_basis="contract"),
            {"radius": "fingerprint_radius", "size": "fingerprint_size"},
        ),
        (
            "discover_fused_butina_clusters",
            ClusterArgs(cutoff=0.6, decision_basis="contract"),
            {"cutoff": "cluster_cutoff"},
        ),
        (
            "embed_representative_conformers",
            EmbedArgs(
                representative_count=6,
                policy="include_singleton_if_available",
                conformers_per_representative=8,
                decision_basis="contract",
            ),
            {
                "representative_count": "representative_count",
                "policy": "representative_policy",
                "conformers_per_representative": (
                    "conformers_per_representative"
                ),
            },
        ),
    ],
)
def test_approved_receipt_fields_track_controller_workflow_keywords(
    stage, arguments, field_to_workflow_keyword
):
    receipt = command_receipt(stage, arguments)
    approved = _approved_call(receipt)
    controller = _controller_mapping(stage)
    inspected_mapping = {
        source.removeprefix("arguments."): workflow_keyword
        for workflow_keyword, source in controller.items()
    }

    assert inspected_mapping == field_to_workflow_keyword
    assert list(_literal_keywords(approved)) == list(field_to_workflow_keyword)
    assert _literal_keywords(approved) == {
        field: getattr(arguments, field) for field in field_to_workflow_keyword
    }
    assert demo_agent._executor_arguments(stage, arguments) == {
        workflow_keyword: getattr(arguments, field)
        for field, workflow_keyword in field_to_workflow_keyword.items()
    }


def test_inspection_receipt_tracks_the_default_executor_call_site():
    tree = _source_tree(demo_agent._default_executors)
    executors = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict))
    entries = {
        ast.literal_eval(key): value for key, value in zip(executors.keys, executors.values)
    }
    actual = entries["inspect_library"].body
    receipt = command_receipt("inspect_library", InspectionArgs())
    displayed = _one_receipt_call(receipt, "inspect_library")

    assert _call_name(actual) == _call_name(displayed) == "inspect_library"
    assert [ast.unparse(value) for value in actual.args] == ["state", "_DATA_PATH"]
    assert [ast.unparse(value) for value in displayed.args] == ["state", "DATA_PATH"]


def test_fingerprint_receipt_tracks_the_real_generator_and_batch_call_sites():
    arguments = FingerprintArgs(radius=3, size=2048, decision_basis="contract")
    receipt = command_receipt("generate_morgan_fingerprints", arguments)
    actual_generator = _one_source_call(
        chemistry_workflow.generate_morgan_fingerprints,
        "_morgan_generator_class",
        keyword_names={"radius", "fpSize"},
    )
    actual_batch = _one_source_call(
        chemistry_workflow.generate_morgan_fingerprints, "GetFingerprints"
    )
    displayed_generator = _one_receipt_call(receipt, "MorganFingerprintGenerator")
    displayed_batch = _one_receipt_call(receipt, "GetFingerprints")
    wrapper_return = _return_expression(chemistry_workflow._morgan_generator_class)

    assert _import_contract(chemistry_workflow._morgan_generator_class) == (
        "nvmolkit.fingerprints",
        ["MorganFingerprintGenerator"],
    )
    assert ast.unparse(wrapper_return) == _call_name(displayed_generator)
    assert isinstance(actual_generator.func, ast.Call)
    assert _keyword_sources(actual_generator) == {
        "radius": "fingerprint_radius",
        "fpSize": "fingerprint_size",
    }
    assert _literal_keywords(displayed_generator) == {
        "radius": arguments.radius,
        "fpSize": arguments.size,
    }
    assert [ast.unparse(value) for value in actual_batch.args] == ["state.molecules"]
    assert [ast.unparse(value) for value in displayed_batch.args] == ["molecules"]


def test_similarity_receipt_tracks_the_real_batch_call_site():
    receipt = command_receipt("measure_tanimoto_similarity", SimilarityArgs())
    actual = _one_source_call(
        chemistry_workflow.measure_tanimoto_similarity,
        "_cross_tanimoto_similarity",
    )
    displayed = _one_receipt_call(receipt, "crossTanimotoSimilarity")
    public = _one_source_call(
        chemistry_workflow._cross_tanimoto_similarity,
        "crossTanimotoSimilarity",
    )

    assert _import_contract(chemistry_workflow._cross_tanimoto_similarity) == (
        "nvmolkit.similarity",
        ["crossTanimotoSimilarity"],
    )
    assert [ast.unparse(value) for value in actual.args] == ["state.fingerprints"]
    assert _call_name(public) == _call_name(displayed)
    assert [ast.unparse(value) for value in public.args] == ["fingerprints"]
    assert [ast.unparse(value) for value in displayed.args] == ["fingerprints"]


def test_cluster_receipt_tracks_the_real_fused_butina_call_site():
    arguments = ClusterArgs(cutoff=0.6, decision_basis="contract")
    receipt = command_receipt("discover_fused_butina_clusters", arguments)
    actual = _one_source_call(
        chemistry_workflow.discover_fused_butina_clusters, "_fused_butina"
    )
    displayed = _one_receipt_call(receipt, "fused_butina")
    public = _one_source_call(chemistry_workflow._fused_butina, "fused_butina")
    cutoff_assignment = _assigned_expression(
        chemistry_workflow.discover_fused_butina_clusters, "cutoff"
    )

    assert _import_contract(chemistry_workflow._fused_butina) == (
        "nvmolkit.clustering",
        ["fused_butina"],
    )
    assert [ast.unparse(value) for value in actual.args] == [
        "state.fingerprints.torch()"
    ]
    assert ast.unparse(cutoff_assignment) == "float(cluster_cutoff)"
    assert _keyword_sources(actual) == {"cutoff": "cutoff"}
    assert _call_name(public) == _call_name(displayed)
    assert [ast.unparse(value) for value in public.args] == ["fingerprints"]
    assert _keyword_sources(public) == {"cutoff": "cutoff"}
    assert [ast.unparse(value) for value in displayed.args] == [
        "fingerprints.torch()"
    ]
    assert _literal_keywords(displayed) == {"cutoff": arguments.cutoff}


def test_embedding_receipt_tracks_the_real_embed_call_site():
    arguments = EmbedArgs(
        representative_count=6,
        policy="include_singleton_if_available",
        conformers_per_representative=8,
        decision_basis="contract",
    )
    receipt = command_receipt("embed_representative_conformers", arguments)
    actual = _one_source_call(
        chemistry_workflow.embed_representative_conformers, "_embed_molecules"
    )
    displayed = _one_receipt_call(receipt, "EmbedMolecules")
    public = _one_source_call(chemistry_workflow._embed_molecules, "EmbedMolecules")
    policy_assignment = _assigned_expression(
        chemistry_workflow.embed_representative_conformers, "policy"
    )
    selection = _one_source_call(
        chemistry_workflow.embed_representative_conformers,
        "select_representatives",
    )

    assert _import_contract(chemistry_workflow._embed_molecules) == (
        "nvmolkit.embedMolecules",
        ["EmbedMolecules"],
    )
    assert [ast.unparse(value) for value in actual.args] == [
        "molecules",
        "parameters",
    ]
    assert ast.unparse(policy_assignment) == (
        "RepresentativePolicy(representative_policy)"
    )
    assert [ast.unparse(value) for value in selection.args] == [
        "state",
        "representative_count",
        "policy",
    ]
    assert _keyword_sources(actual) == {
        "confsPerMolecule": "conformers_per_representative",
        "maxIterations": "-1",
    }
    assert _call_name(public) == _call_name(displayed)
    assert [ast.unparse(value) for value in public.args] == [
        "molecules",
        "parameters",
    ]
    assert _keyword_sources(public) == {
        "confsPerMolecule": "confsPerMolecule",
        "maxIterations": "maxIterations",
    }
    assert [ast.unparse(value) for value in displayed.args] == [
        "molecules",
        "parameters",
    ]
    assert _literal_keywords(displayed) == {
        "confsPerMolecule": arguments.conformers_per_representative,
        "maxIterations": -1,
    }


def test_optimization_receipt_tracks_the_real_mmff94_call_site():
    receipt = command_receipt("optimize_conformers_mmff94", OptimizationArgs())
    actual = _one_source_call(
        chemistry_workflow.optimize_conformers_mmff94, "_optimize_mmff94"
    )
    displayed = _one_receipt_call(receipt, "MMFFOptimizeMoleculesConfs")
    public = _one_source_call(
        chemistry_workflow._optimize_mmff94, "MMFFOptimizeMoleculesConfs"
    )
    coordinate_return = _return_expression(
        chemistry_workflow._coordinate_output_device
    )

    assert _import_contract(chemistry_workflow._optimize_mmff94) == (
        "nvmolkit.mmffOptimization",
        ["MMFFOptimizeMoleculesConfs"],
    )
    assert _import_contract(chemistry_workflow._coordinate_output_device) == (
        "nvmolkit.types",
        ["CoordinateOutput"],
    )
    assert ast.unparse(coordinate_return) == "CoordinateOutput.DEVICE"
    assert [ast.unparse(value) for value in actual.args] == ["molecules"]
    assert _keyword_sources(actual) == {
        "maxIters": "500",
        "output": "_coordinate_output_device()",
    }
    assert _call_name(public) == _call_name(displayed)
    assert [ast.unparse(value) for value in public.args] == ["molecules"]
    assert _keyword_sources(public) == {
        "maxIters": "maxIters",
        "output": "output",
    }
    assert [ast.unparse(value) for value in displayed.args] == ["molecules"]
    assert _keyword_sources(displayed) == {
        "maxIters": "500",
        "output": "CoordinateOutput.DEVICE",
    }
