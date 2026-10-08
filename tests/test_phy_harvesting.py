import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from chiplet_actuary.harvesting import HarvestingDie, Resource, Tier, family_cost
from chiplet_actuary.harvesting_cli import make_die
from test_harvesting import brute_probability


def physical_die():
    return HarvestingDie(
        "phy_test",
        "5",
        100,
        Resource("gpu", 2, 40, 2),
        Resource("cpu", 2, 20, 1),
        Resource("l2", 2, 10, 1),
        Resource("l3", 2, 10, 1),
        Resource("pcie_phys", 2, 10, 1),
        Resource("memory_phys", 2, 20, 128),
    )


def requirements():
    return [Tier("L", 4, 2, 2, 2, 100, 2, 256), Tier("S", 2, 1, 1, 1, 100, 1, 128)]


class PhyHarvestingTests(unittest.TestCase):
    def test_six_pool_integral_matches_enumeration(self):
        die = physical_die()
        r = die.yields(requirements())
        self.assertEqual(r["eligible_yields"]["L"], die.upstream_chip.die_yield())
        self.assertAlmostEqual(
            r["eligible_yields"]["S"], brute_probability(die, (1,) * 6), places=8
        )
        self.assertEqual(r["required_blocks"]["L"], (2,) * 6)
        self.assertAlmostEqual(
            sum(r["exclusive_bin_yields"].values()) + r["scrap_yield"], 1
        )

    def test_phy_only_damage_salvage_and_both_requirements(self):
        die = physical_die()
        high = requirements()[0]
        pcie = replace(high, name="pcie_cut", pcie_x8_count=1)
        memory = replace(high, name="memory_cut", memory_bus_bits=128)
        both = replace(pcie, name="both_cut", memory_bus_bits=128)
        full = die.upstream_chip.die_yield()

        def probability(t):
            return die.yields([t])["eligible_yields"][t.name]

        self.assertGreater(probability(pcie), full)
        self.assertGreater(probability(memory), full)
        self.assertGreater(
            probability(both), max(probability(pcie), probability(memory))
        )

    def test_fixed_granularity_and_missing_requirements(self):
        die = physical_die()
        for bits in (64, 129, True):
            with self.assertRaises(ValueError):
                replace(requirements()[0], memory_bus_bits=bits)
        with self.assertRaises(ValueError):
            replace(die, memory_phys=Resource("memory", 4, 20, 64))
        with self.assertRaises(ValueError):
            replace(die, pcie_phys=Resource("pcie", 2, 10, 2))
        with self.assertRaises(ValueError):
            die.yields([Tier("x", 4, 2, 2, 2, 1)])
        with self.assertRaises(ValueError):
            replace(die, pcie_phys=None, memory_phys=None).yields(requirements())
        with self.assertRaises(ValueError):
            die.yields([requirements()[0], replace(requirements()[1], pcie_x8_count=3)])

    def test_disabled_harvesting_cost_reproduces_upstream(self):
        result = family_cost(
            physical_die(), requirements(), harvesting=False, whole_wafers=False
        )
        self.assertAlmostEqual(
            result["average_manufacturing_per_shipped_unit"],
            result["upstream_unharvested_manufacturing_per_unit"],
            places=9,
        )

    def test_example_io_partition_and_four_edges(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "examples/harvesting_5nm.json").read_text())
        die, area = make_die(config)
        self.assertTrue(area["valid"])
        self.assertEqual([r.count for r in die.resources][-2:], [4, 4])
        self.assertAlmostEqual(die.area, area["area_breakdown_mm2"]["total"])
        old_config = dict(config, harvest_io_phys=False)
        mandatory_die, _ = make_die(old_config)
        self.assertAlmostEqual(die.area, mandatory_die.area)
        self.assertAlmostEqual(
            mandatory_die.mandatory_area_mm2 - die.mandatory_area_mm2,
            die.pcie_phys.total_area_mm2 + die.memory_phys.total_area_mm2,
        )
        shore = area["constraints"]["io_phy_shoreline"]
        self.assertEqual(shore["edge_count"], 4)
        self.assertAlmostEqual(shore["available_length_mm"], 4 * math.sqrt(die.area))
        self.assertAlmostEqual(
            shore["required_length_mm"], 46.204081632653065 + (47 + 422 + 64) * 26 / 645
        )
        from area_model.soc_model import calculate_soc_area

        two = calculate_soc_area(**dict(config["soc"], phy_edge_count=2))
        self.assertFalse(two["valid"])
        self.assertEqual(two["area_breakdown_mm2"], area["area_breakdown_mm2"])
        for edge in (0, 3, True):
            with self.assertRaises(ValueError):
                calculate_soc_area(**dict(config["soc"], phy_edge_count=edge))

    def test_micropod_cannot_override_two_edges(self):
        workspace = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(workspace))
        from cost_model.micropod import estimate_micropod

        c = json.loads(
            (workspace / "cost_model/examples/micropod_5nm.json").read_text()
        )
        self.assertEqual(
            estimate_micropod(c)["soc_area_result"]["constraints"]["io_phy_shoreline"][
                "edge_count"
            ],
            2,
        )
        c["soc"]["phy_edge_count"] = 4
        with self.assertRaises(ValueError):
            estimate_micropod(c)


if __name__ == "__main__":
    unittest.main()
