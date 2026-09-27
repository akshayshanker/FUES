"""Estimation specs agree with the calibration the solver actually uses."""
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kikku.run.estimate import load_estimation_spec
from examples.durables.solve import load_spec, make_spec

SYNTAX = REPO / "examples" / "durables" / "syntax"
CASES = [
    (reg, spec)
    for reg in sorted(p for p in SYNTAX.iterdir() if p.is_dir())
    for spec in sorted((reg / "estimation").glob("*.yaml"))
]


def _case_id(value):
    return value.name if isinstance(value, Path) else str(value)


def resolved_calibration_keys(registry, spec_factory_name):
    """Keys of the first-stage calibration that solve() builds from make_spec."""
    recipe = load_spec(str(registry / spec_factory_name))
    spec = make_spec(recipe, registry_dir=str(registry))
    stage_names = list(recipe.stages.keys())
    return set(spec[stage_names[0]][0]["calibration"].keys())


def types_block(raw):
    return (raw.get("estimation") or {}).get("types") or {}


@pytest.mark.parametrize("registry,spec_path", CASES, ids=_case_id)
def test_free_parameters_are_model_parameters(registry, spec_path):
    spec = load_estimation_spec(str(spec_path))
    raw = yaml.safe_load(spec_path.read_text())
    factory = (
        "spec_factory_males.yaml"
        if spec_path.stem.endswith("_males")
        else "spec_factory.yaml"
    )
    keys = resolved_calibration_keys(registry, factory)
    mapped = set()
    parameters = (types_block(raw).get("parameters") or {})
    for target, blk in parameters.items():
        assert target in keys and target not in spec["free"]
        mapped |= {blk["location"], blk["spread"]}
        assert not (mapped & keys)
    for name in spec["free"]:
        if name in mapped:
            continue
        assert name in keys, (
            f"{spec_path.name}: {name} is not a parameter of {registry.name}"
        )


@pytest.mark.parametrize("registry,spec_path", CASES, ids=_case_id)
def test_targets_exist_in_data(registry, spec_path):
    spec = load_estimation_spec(str(spec_path))
    if spec["moment_spec"].get("data_source") != "precomputed":
        pytest.skip("no data file")
    keys = set(spec["data_moments"])
    for t in spec["moment_spec"].get("targets") or []:
        assert any(
            k == t["key"] or k.startswith(t["key"] + "__") for k in keys
        ), t["key"]


def test_cobb_douglas_uses_rho():
    for p in (SYNTAX / "cobb_douglas" / "estimation").glob("baseline*.yaml"):
        free = load_estimation_spec(str(p))["free"]
        assert (
            "rho" in free
            and "gamma_c" not in free
            and "gamma_h" not in free
            and "theta" in free
        ), p.name
