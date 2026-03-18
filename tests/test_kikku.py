"""Tests for kikku period graph construction and topology derivation."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

import yaml
from pathlib import Path
from dolo.compiler.model import SymbolicModel
from kikku.dynx import period_to_graph, backward_paths, forward_order
from kikku.dynx import load_inter_connector

SYNTAX_DIR = Path(REPO_ROOT) / "examples" / "retirement" / "syntax"


def _build_period_dict():
    """Build a minimal period dict from the retirement syntax."""
    stages_dir = SYNTAX_DIR / "stages"
    with open(SYNTAX_DIR / "period.yaml") as f:
        raw = yaml.safe_load(f)

    stage_names = []
    for entry in raw.get("stages", []):
        if isinstance(entry, dict):
            stage_names.extend(entry.keys())
        else:
            stage_names.append(str(entry))

    stages = {}
    for name in stage_names:
        path = stages_dir / name / f"{name}.yaml"
        with open(path) as f:
            stages[name] = SymbolicModel(
                yaml.compose(f.read()), filename=str(path),
            )

    return {"stages": stages, "connectors": []}


class TestPeriodToGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.period = _build_period_dict()
        cls.graph = period_to_graph(cls.period)

    def test_nodes(self):
        expected = {"labour_mkt_decision", "work_cons", "retire_cons"}
        self.assertEqual(set(self.graph.nodes), expected)

    def test_edges(self):
        edges = list(self.graph.edges)
        self.assertIn(("labour_mkt_decision", "work_cons"), edges)
        self.assertIn(("labour_mkt_decision", "retire_cons"), edges)
        self.assertEqual(len(edges), 2)

    def test_edge_branch_attrs(self):
        d_work = self.graph.edges["labour_mkt_decision", "work_cons"]
        d_ret = self.graph.edges["labour_mkt_decision", "retire_cons"]
        self.assertEqual(d_work["branch"], "work")
        self.assertEqual(d_ret["branch"], "retire")
        self.assertEqual(d_work["rename"], {})
        self.assertEqual(d_ret["rename"], {})

    def test_node_interface_attrs(self):
        lmkt = self.graph.nodes["labour_mkt_decision"]
        self.assertEqual(lmkt["prestate"], ["a"])
        self.assertEqual(lmkt["kind"], "branching")
        self.assertIn("a", lmkt["poststate"])
        self.assertIn("a_ret", lmkt["poststate"])

        wc = self.graph.nodes["work_cons"]
        self.assertEqual(wc["prestate"], ["a"])
        self.assertEqual(wc["poststate"], ["b"])

        rc = self.graph.nodes["retire_cons"]
        self.assertEqual(rc["prestate"], ["a_ret"])
        self.assertEqual(rc["poststate"], ["b_ret"])


class TestBackwardPaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.period = _build_period_dict()
        cls.graph = period_to_graph(cls.period)
        cls.inter_conn = load_inter_connector(SYNTAX_DIR)

    def test_inter_connector(self):
        self.assertEqual(self.inter_conn, {"b": "a", "b_ret": "a_ret"})

    def test_wave_structure(self):
        waves = backward_paths(self.graph, self.inter_conn)
        self.assertEqual(len(waves), 2)
        self.assertIn("retire_cons", waves[0])
        self.assertIn("work_cons", waves[0])
        self.assertEqual(waves[1], ["labour_mkt_decision"])

    def test_all_stages_covered(self):
        waves = backward_paths(self.graph, self.inter_conn)
        all_stages = {s for wave in waves for s in wave}
        self.assertEqual(
            all_stages,
            {"labour_mkt_decision", "work_cons", "retire_cons"},
        )

    def test_merge_after_leaves(self):
        """labour_mkt_decision depends on both leaf stages."""
        waves = backward_paths(self.graph, self.inter_conn)
        leaf_wave = waves[0]
        merge_wave = waves[1]
        self.assertIn("work_cons", leaf_wave)
        self.assertIn("retire_cons", leaf_wave)
        self.assertEqual(merge_wave, ["labour_mkt_decision"])


class TestForwardOrder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.period = _build_period_dict()
        cls.graph = period_to_graph(cls.period)

    def test_forward_order(self):
        fwd = forward_order(self.graph)
        self.assertEqual(fwd[0], "labour_mkt_decision")
        self.assertEqual(set(fwd[1:]), {"work_cons", "retire_cons"})


if __name__ == "__main__":
    unittest.main()
