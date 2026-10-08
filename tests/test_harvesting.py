import itertools
import math
import unittest
from unittest.mock import patch

from chiplet_actuary import spec
from chiplet_actuary.harvesting import (
    HarvestingDie,
    Resource,
    Tier,
    allocate_supply,
    family_cost,
)
from chiplet_actuary.harvesting_adapter import from_soc_area


def example():
    return HarvestingDie(
        "small",
        "7",
        40,
        Resource("gpu_tpcs", 4, 80, 2),
        Resource("cpu_cores", 4, 20, 1),
        Resource("gpu_l2_slices", 4, 12, 2),
        Resource("cpu_l3_slices", 4, 8, 1),
    )


def tiers():
    return [
        Tier("L", 8, 4, 8, 4, 1000),
        Tier("M", 4, 2, 4, 2, 1000),
        Tier("S", 2, 1, 2, 1, 1000),
    ]


def brute_probability(die, minimum):
    """Independent finite inclusion-exclusion enumeration, no quadrature/binom.sf."""
    density = spec.Defect_Density_Die[die.node] / 100
    shape = spec.critical_level

    def good(area):
        return (1 + density * area / shape) ** (-shape)

    result = 0.0
    for survivors in itertools.product(
        *(range(k, r.count + 1) for k, r in zip(minimum, die.resources))
    ):
        multiplicity = math.prod(
            math.comb(r.count, g) for r, g in zip(die.resources, survivors)
        )
        good_area = die.mandatory_area_mm2 + sum(
            r.block_area * g for r, g in zip(die.resources, survivors)
        )
        bad = [r.count - g for r, g in zip(die.resources, survivors)]
        probability = 0.0
        for included in itertools.product(*(range(n + 1) for n in bad)):
            coefficient = (-1) ** sum(included) * math.prod(
                math.comb(n, j) for n, j in zip(bad, included)
            )
            area = good_area + sum(
                r.block_area * j for r, j in zip(die.resources, included)
            )
            probability += coefficient * good(area)
        result += multiplicity * probability
    return result


class HarvestingTests(unittest.TestCase):
    def test_full_yield_exactly_matches_upstream(self):
        die = example()
        result = die.yields(tiers())
        self.assertEqual(result["eligible_yields"]["L"], die.upstream_chip.die_yield())

    def test_bins_are_disjoint_and_conserve_probability(self):
        result = example().yields(tiers())
        bins = list(result["exclusive_bin_yields"].values())
        self.assertTrue(all(p >= 0 for p in bins))
        self.assertAlmostEqual(sum(bins) + result["scrap_yield"], 1)
        self.assertGreater(result["eligible_yields"]["S"], result["full_die_yield"])

    def test_quadrature_matches_independent_enumeration(self):
        die = HarvestingDie(
            "tiny",
            "7",
            100,
            Resource("gpu", 2, 50, 2),
            Resource("cpu", 2, 30, 1),
            Resource("l2", 2, 20, 1),
            Resource("l3", 2, 10, 1),
        )
        result = die.yields([Tier("low", 2, 1, 1, 1, 1)])
        self.assertAlmostEqual(
            result["eligible_yields"]["low"],
            brute_probability(die, (1, 1, 1, 1)),
            places=9,
        )

    def test_public_region_limits_salvage(self):
        die = example()
        density = spec.Defect_Density_Die["7"] / 100
        public_yield = (1 + density * die.mandatory_area_mm2 / spec.critical_level) ** (
            -spec.critical_level
        )
        self.assertLessEqual(die.yields(tiers())["eligible_yields"]["S"], public_yield)

    def test_no_defects_all_full_spec(self):
        with patch.dict(spec.Defect_Density_Die, {"7": 0}):
            result = example().yields(tiers())
        self.assertEqual(result["exclusive_bin_yields"], {"L": 1, "M": 0, "S": 0})
        self.assertEqual(result["scrap_yield"], 0)

    def test_cache_damage_can_be_harvested(self):
        die = example()
        ts = [Tier("full", 8, 4, 8, 4, 1), Tier("cache_cut", 8, 4, 4, 2, 1)]
        self.assertGreater(die.yields(ts)["exclusive_bin_yields"]["cache_cut"], 0)

    def test_cpu_and_gpu_both_required(self):
        die = example()
        gpu_cut = die.yields([Tier("x", 4, 4, 8, 4, 1)])["eligible_yields"]["x"]
        both_cut = die.yields([Tier("x", 4, 2, 8, 4, 1)])["eligible_yields"]["x"]
        self.assertGreater(both_cut, gpu_cut)

    def test_harvesting_disabled_matches_original_cost(self):
        die = example()
        result = family_cost(die, tiers(), harvesting=False, whole_wafers=False)
        self.assertAlmostEqual(
            result["average_manufacturing_per_shipped_unit"],
            result["upstream_unharvested_manufacturing_per_unit"],
            places=9,
        )
        self.assertEqual(result["yields"]["exclusive_bin_yields"]["M"], 0)

    def test_downgrading_and_no_upgrading(self):
        matrix, unused = allocate_supply([10, 10, 10], [5, 10, 12])
        self.assertEqual(matrix, [[5, 0, 2], [0, 10, 0], [0, 0, 10]])
        self.assertEqual(unused, [3, 0, 0])
        with self.assertRaises(ArithmeticError):
            allocate_supply([0, 100], [1, 0])

    def test_supply_conservation_and_nre_once(self):
        die = example()
        result = family_cost(die, tiers(), whole_wafers=False)
        base = family_cost(die, tiers(), whole_wafers=False, harvesting=False)
        production = result["production"]
        allocated = sum(
            sum(row.values()) for row in production["allocation_source_to_sku"].values()
        )
        self.assertAlmostEqual(allocated, sum(production["prepackage_demand"].values()))
        self.assertAlmostEqual(
            production["fabricated_dies"],
            allocated
            + sum(production["unused_bin_supply"].values())
            + production["scrap_dies"],
        )
        self.assertEqual(result["nre_cost"], base["nre_cost"])
        self.assertLess(result["manufacturing_total"], base["manufacturing_total"])
        self.assertEqual(result["physical_die_area_mm2"], base["physical_die_area_mm2"])

    def test_high_only_demand_gets_no_salvage_credit(self):
        ts = [
            Tier("L", 8, 4, 8, 4, 1000),
            Tier("M", 4, 2, 4, 2, 0),
            Tier("S", 2, 1, 2, 1, 0),
        ]
        enabled = family_cost(example(), ts, whole_wafers=False)
        disabled = family_cost(example(), ts, harvesting=False, whole_wafers=False)
        self.assertAlmostEqual(
            enabled["manufacturing_total"], disabled["manufacturing_total"]
        )
        self.assertGreater(sum(enabled["production"]["unused_bin_supply"].values()), 0)

    def test_whole_wafer_option(self):
        result = family_cost(example(), tiers())
        wafers = result["production"]["wafers"]
        self.assertEqual(wafers, int(wafers))

    def test_fixed_granularity_and_bad_inputs(self):
        with self.assertRaises(ValueError):
            Tier("bad", 3, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            Resource("cache", 0, 1, 1)
        with self.assertRaises(ValueError):
            HarvestingDie(
                "bad", "7", 1, Resource("gpu", 2, 1, 1), *example().resources[1:]
            )
        with self.assertRaises(ValueError):
            example().yields([Tier("x", 10, 4, 8, 4, 1)])
        with self.assertRaises(ValueError):
            example().yields(list(reversed(tiers())))
        with self.assertRaises(ValueError):
            example().yields([tiers()[0], tiers()[0]])

    def test_minimum_cache_rounded_up_to_slice(self):
        result = example().yields([Tier("x", 2, 1, 3, 1.5, 1)])
        self.assertEqual(result["required_blocks"]["x"], (1, 1, 2, 2))

    def test_adapter_partition_and_failed_geometry(self):
        result = {
            "valid": True,
            "config": {"physical_sm_count": 8, "cpu_core_count": 4, "l2_size_mb": 8},
            "area_breakdown_mm2": {
                "total": 200,
                "components": {"cpu_cores_and_private_cache": 20, "cpu_shared_l3": 8},
            },
            "gpu": {
                "area_breakdown_mm2": {
                    "sm_total": 40,
                    "extra_gpc_tpc": 10,
                    "l2_total": 14,
                }
            },
        }
        kwargs = {
            "name": "adapter",
            "node": "7",
            "gpu_l2_slice_count": 4,
            "cpu_l3_slice_count": 4,
            "cpu_l3_total_mb": 4,
            "gpu_tpc_extra_area_mm2": 4,
            "gpu_l2_shared_control_area_mm2": 2,
        }
        die = from_soc_area(result, **kwargs)
        self.assertEqual(die.area, 200)
        self.assertEqual(die.mandatory_area_mm2, 116)
        result["valid"] = False
        with self.assertRaises(ValueError):
            from_soc_area(result, **kwargs)


if __name__ == "__main__":
    unittest.main()
