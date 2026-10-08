"""Defect harvesting for ONE physical monolithic die and a nested SKU family.

Conditional Poisson defects share a Gamma-distributed intensity across the die.
This exactly recovers the upstream negative-binomial all-good probability.
SciPy quadrature is used for bin probabilities, not independent block NB yields.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import binom

from chiplet_actuary import spec
from chiplet_actuary.chip import Chip
from chiplet_actuary.module import Module
from chiplet_actuary.package import OS


def _nonnegative(name, value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def _integer(name, value, positive=False):
    _nonnegative(name, value, positive)
    if int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Resource:
    name: str
    count: int
    total_area_mm2: float
    units_per_block: float

    def __post_init__(self):
        _integer(f"{self.name}.count", self.count, positive=True)
        _nonnegative(f"{self.name}.area", self.total_area_mm2, positive=True)
        _nonnegative(
            f"{self.name}.units_per_block", self.units_per_block, positive=True
        )

    @property
    def block_area(self):
        return self.total_area_mm2 / self.count

    def required_blocks(self, units):
        _nonnegative(f"{self.name} requirement", units, positive=True)
        value = units / self.units_per_block
        nearest = round(value)
        if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12):
            value = nearest
        required = math.ceil(value)
        if required > self.count:
            raise ValueError(f"{self.name} requirement exceeds manufactured resources")
        return required


@dataclass(frozen=True)
class Tier:
    name: str
    gpu_sms: int
    cpu_cores: int
    gpu_l2_mb: float
    cpu_l3_mb: float
    demand: int
    pcie_x8_count: Optional[int] = None
    memory_bus_bits: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Tier name must be nonempty")
        if _integer("gpu_sms", self.gpu_sms, positive=True) % 2:
            raise ValueError("GPU requirements must be multiples of a fixed 2-SM TPC")
        _integer("cpu_cores", self.cpu_cores, positive=True)
        _nonnegative("gpu_l2_mb", self.gpu_l2_mb, positive=True)
        _nonnegative("cpu_l3_mb", self.cpu_l3_mb, positive=True)
        _integer("demand", self.demand)
        if self.pcie_x8_count is not None:
            _integer("pcie_x8_count", self.pcie_x8_count, positive=True)
        if self.memory_bus_bits is not None:
            if _integer("memory_bus_bits", self.memory_bus_bits, positive=True) % 128:
                raise ValueError(
                    "Memory PHY requirements must be multiples of 128 bits"
                )

    @property
    def requirements(self):
        return (self.gpu_sms, self.cpu_cores, self.gpu_l2_mb, self.cpu_l3_mb)


@dataclass(frozen=True)
class HarvestingDie:
    name: str
    node: str
    mandatory_area_mm2: float
    gpu_tpcs: Resource
    cpu_cores: Resource
    gpu_l2_slices: Resource
    cpu_l3_slices: Resource
    pcie_phys: Optional[Resource] = None
    memory_phys: Optional[Resource] = None

    def __post_init__(self):
        if self.node not in spec.Defect_Density_Die:
            raise ValueError(f"Unsupported node {self.node}; choose an upstream node")
        _nonnegative("mandatory_area_mm2", self.mandatory_area_mm2)
        if self.gpu_tpcs.units_per_block != 2:
            raise ValueError("GPU granularity is fixed at 2 SM per TPC")
        if self.cpu_cores.units_per_block != 1:
            raise ValueError("CPU granularity is fixed at 1 core")
        if self.pcie_phys is not None and self.pcie_phys.units_per_block != 1:
            raise ValueError("PCIe granularity is fixed at one x8 PHY")
        if self.memory_phys is not None and self.memory_phys.units_per_block != 128:
            raise ValueError("Memory PHY granularity is fixed at 128 bits")

    @property
    def resources(self):
        base = (self.gpu_tpcs, self.cpu_cores, self.gpu_l2_slices, self.cpu_l3_slices)
        return base + tuple(
            r for r in (self.pcie_phys, self.memory_phys) if r is not None
        )

    def _tier_requirements(self, tier):
        values = list(tier.requirements)
        for pool, amount in (
            (self.pcie_phys, tier.pcie_x8_count),
            (self.memory_phys, tier.memory_bus_bits),
        ):
            if pool is None:
                if amount is not None:
                    raise ValueError(
                        "PHY tier requirement provided without a physical PHY resource pool"
                    )
            else:
                if amount is None:
                    raise ValueError(
                        "Each tier must specify requirements for all physical PHY pools"
                    )
                values.append(amount)
        return values

    @property
    def area(self):
        return self.mandatory_area_mm2 + sum(r.total_area_mm2 for r in self.resources)

    @cached_property
    def upstream_chip(self):
        # One module abstraction for NRE: no repeated design charge per core/slice/SKU.
        module = Module(self.name + "_design", self.node, self.area)
        return Chip(self.name, self.node, {module: 1})

    def _thresholds(self, tiers):
        if not tiers or len({t.name for t in tiers}) != len(tiers):
            raise ValueError("Require nonempty tiers with unique names, highest first")
        thresholds = []
        previous = None
        for tier in tiers:
            required = tuple(
                r.required_blocks(u)
                for r, u in zip(self.resources, self._tier_requirements(tier))
            )
            if previous is not None and any(a > b for a, b in zip(required, previous)):
                raise ValueError(
                    "Tiers must be nested, highest to lowest, across all resources"
                )
            thresholds.append(required)
            previous = required
        return thresholds

    def _eligible_probability(self, required):
        density = spec.Defect_Density_Die[self.node] / 100  # defects/mm^2
        shape = spec.critical_level
        _nonnegative("defect_density", density)
        _nonnegative("critical_level", shape, positive=True)
        if density == 0:
            return 1.0
        if all(k == r.count for k, r in zip(required, self.resources)):
            return self.upstream_chip.die_yield()

        def integrand(x):
            if x <= 0:
                return 0.0
            # X~Gamma(shape, rate=shape), E[X]=1. Same X for every die region.
            log_pdf = (
                shape * math.log(shape)
                - gammaln(shape)
                + (shape - 1) * math.log(x)
                - shape * x
            )
            probability = math.exp(log_pdf - density * x * self.mandatory_area_mm2)
            for resource, minimum in zip(self.resources, required):
                survives = math.exp(-density * x * resource.block_area)
                probability *= binom.sf(minimum - 1, resource.count, survives)
            return probability

        value, error = quad(
            integrand, 0, math.inf, epsabs=1e-11, epsrel=1e-9, limit=200
        )
        if not math.isfinite(value) or error > max(1e-9, abs(value) * 1e-7):
            raise ArithmeticError("Harvesting yield integration did not converge")
        return min(1.0, max(0.0, value))

    def yields(self, tiers: List[Tier], harvesting=True):
        if not isinstance(harvesting, bool):
            raise ValueError("harvesting must be boolean")
        thresholds = self._thresholds(tiers)
        full = self.upstream_chip.die_yield()
        eligible = [
            self._eligible_probability(k) if harvesting else full for k in thresholds
        ]
        bins = []
        previous = 0.0
        for probability in eligible:
            if probability < previous - 1e-9:
                raise ArithmeticError("Eligible probabilities must be nondecreasing")
            bins.append(max(0.0, probability - previous))
            previous = max(previous, probability)
        return {
            "full_die_yield": full,
            "eligible_yields": dict(zip((t.name for t in tiers), eligible)),
            "exclusive_bin_yields": dict(zip((t.name for t in tiers), bins)),
            "scrap_yield": 1 - sum(bins),
            "required_blocks": dict(zip((t.name for t in tiers), thresholds)),
            "resource_order": [r.name for r in self.resources],
            "harvesting_enabled": harvesting,
        }


def allocate_supply(bin_supply, demands):
    """High-to-low demand; consume the lowest eligible bin first. No upgrades."""
    remaining = list(bin_supply)
    matrix = [[0.0] * len(demands) for _ in demands]  # source bin x destination SKU
    for destination, demand in enumerate(demands):
        needed = demand
        for source in range(destination, -1, -1):
            take = min(remaining[source], needed)
            matrix[source][destination] += take
            remaining[source] -= take
            needed -= take
        if needed > max(1e-7, demand * 1e-9):
            raise ArithmeticError("Insufficient eligible supply for demand")
    return matrix, remaining


def family_cost(
    die: HarvestingDie, tiers: List[Tier], *, harvesting=True, whole_wafers=True
):
    """Expected-value manufacturing plan for one die and one shared OS package.

    No per-bin KGD cost: all fabricated die costs are charged once to the family.
    Packaging loss is replenished per SKU before wafer demand is computed.
    Unused good dies get zero salvage credit. Integer wafers do not guarantee
    realized stochastic demand fulfillment; this is an expectation model.
    """
    if not isinstance(whole_wafers, bool):
        raise ValueError("whole_wafers must be boolean")
    yields = die.yields(tiers, harvesting)
    demand = [t.demand for t in tiers]
    if not sum(demand):
        raise ValueError("At least one SKU must have positive demand")
    chip = die.upstream_chip
    package = OS(die.name + "_shared_package", {chip: 1})
    package_yield = spec.bonding_yield_os
    if not 0 < package_yield <= 1:
        raise ValueError("Invalid package yield")
    die_per_wafer = chip.N_die_total()
    if die_per_wafer <= 0:
        raise ValueError("Die area exceeds upstream wafer model range")
    inputs = [n / package_yield for n in demand]
    probabilities = [yields["exclusive_bin_yields"][t.name] for t in tiers]
    needed_die = 0.0
    for i in range(len(tiers)):
        required, eligible = sum(inputs[: i + 1]), sum(probabilities[: i + 1])
        if required and eligible <= 0:
            return {
                "feasible": False,
                "reason": f"No eligible supply for {tiers[i].name}",
                "yields": yields,
            }
        if required:
            needed_die = max(needed_die, required / eligible)
    wafers = needed_die / die_per_wafer
    wafers = math.ceil(wafers) if whole_wafers else wafers
    fabricated = wafers * die_per_wafer
    supply = [fabricated * p for p in probabilities]
    allocation, unused = allocate_supply(supply, inputs)
    manufacturing = {
        "wafers": wafers * spec.Cost_Wafer_Die[die.node],
        "bumping": sum(inputs) * die.area * spec.c4_bump_cost_factor,
        "raw_packages": sum(inputs) * package.cost_raw_package(),
    }
    # Shared module, die and package designs are charged once across all SKUs.
    nre = {
        "module_design": sum(m.NRE() for m in chip.modules),
        "die_design": chip.NRE(),
        "package_design": package.NRE(),
    }
    re_total, nre_total = sum(manufacturing.values()), sum(nre.values())
    return {
        "feasible": True,
        "scope": "one monolithic physical die, one shared organic-substrate package; expected-value production",
        "node": die.node,
        "physical_die_area_mm2": die.area,
        "yields": yields,
        "production": {
            "whole_wafers": whole_wafers,
            "wafers": wafers,
            "gross_dies_per_wafer": die_per_wafer,
            "fabricated_dies": fabricated,
            "package_yield": package_yield,
            "expected_package_failures": sum(inputs) - sum(demand),
            "demand": dict(zip((t.name for t in tiers), demand)),
            "prepackage_demand": dict(zip((t.name for t in tiers), inputs)),
            "bin_supply": dict(zip((t.name for t in tiers), supply)),
            "unused_bin_supply": dict(zip((t.name for t in tiers), unused)),
            "allocation_source_to_sku": {
                t.name: dict(zip((s.name for s in tiers), row))
                for t, row in zip(tiers, allocation)
            },
            "scrap_dies": fabricated * yields["scrap_yield"],
        },
        "manufacturing_cost": manufacturing,
        "manufacturing_total": re_total,
        "nre_cost": nre,
        "nre_total": nre_total,
        "family_total": re_total + nre_total,
        "average_manufacturing_per_shipped_unit": re_total / sum(demand),
        "average_total_per_shipped_unit": (re_total + nre_total) / sum(demand),
        "upstream_unharvested_manufacturing_per_unit": package.cost_total_system(),
        "notes": [
            "Full die area is manufactured for every SKU; disabling resources never reduces wafer area.",
            "Surplus good dies have no sales or inventory credit in this production campaign.",
            "Package losses consume additional graded dies; no second die-yield charge is applied.",
            "Per-SKU selling prices and accounting cost allocations are not modeled.",
        ],
    }
