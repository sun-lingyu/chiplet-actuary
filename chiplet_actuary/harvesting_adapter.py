"""Map a MicroPodSim SoC area result into harvestable and mandatory regions."""

import math

from chiplet_actuary.harvesting import HarvestingDie, Resource, _integer, _nonnegative


def from_soc_area(
    result,
    *,
    name,
    node,
    gpu_l2_slice_count,
    cpu_l3_slice_count,
    cpu_l3_total_mb,
    gpu_tpc_extra_area_mm2,
    gpu_l2_shared_control_area_mm2,
    harvest_io_phys=False,
):
    """All residual overhead, command front and shared GPC/control logic are mandatory.

    Cache slice area includes the identified local data/control region. Global
    CPU-L3 controls are unidentified and assumed part of mandatory system overhead.
    CPU-L3 total capacity is an explicit input, not inferred from area or image.
    This changes cost technology parameters only; it does not port geometry.
    """
    if result.get("valid", True) is not True:
        raise ValueError(
            "Cannot cost an area configuration that failed its constraints"
        )
    config = result["config"]
    components = result["area_breakdown_mm2"]["components"]
    total = result["area_breakdown_mm2"]["total"]
    gpu = result["gpu"]["area_breakdown_mm2"]
    sms = _integer("physical_sm_count", config["physical_sm_count"], positive=True)
    cores = _integer("cpu_core_count", config["cpu_core_count"], positive=True)
    if sms % 2:
        raise ValueError("Physical GPU must contain whole 2-SM TPCs")
    l2_count = _integer("gpu_l2_slice_count", gpu_l2_slice_count, positive=True)
    l3_count = _integer("cpu_l3_slice_count", cpu_l3_slice_count, positive=True)
    _nonnegative("cpu_l3_total_mb", cpu_l3_total_mb, positive=True)
    _nonnegative("gpu_tpc_extra_area_mm2", gpu_tpc_extra_area_mm2)
    _nonnegative("gpu_l2_shared_control_area_mm2", gpu_l2_shared_control_area_mm2)
    if gpu_tpc_extra_area_mm2 > gpu["extra_gpc_tpc"]:
        raise ValueError("TPC-only overhead exceeds combined GPC/TPC overhead")
    l2_data = gpu["l2_total"] - gpu_l2_shared_control_area_mm2
    resources = (
        Resource("gpu_tpcs", sms // 2, gpu["sm_total"] + gpu_tpc_extra_area_mm2, 2),
        Resource("cpu_cores", cores, components["cpu_cores_and_private_cache"], 1),
        Resource("gpu_l2_slices", l2_count, l2_data, config["l2_size_mb"] / l2_count),
        Resource(
            "cpu_l3_slices",
            l3_count,
            components["cpu_shared_l3"],
            cpu_l3_total_mb / l3_count,
        ),
    )
    if not isinstance(harvest_io_phys, bool):
        raise ValueError("harvest_io_phys must be boolean")
    if harvest_io_phys:
        pcie = _integer("pcie_x8_count", config["pcie_x8_count"], positive=True)
        memory = _integer("memory_bus_bits", config["memory_bus_bits"], positive=True)
        if memory % 128:
            raise ValueError(
                "Physical memory interface must contain whole 128-bit PHY groups"
            )
        resources += (
            Resource("pcie_phys", pcie, components["pcie_phy"], 1),
            Resource("memory_phys", memory // 128, components["memory_phy"], 128),
        )
    mandatory = total - sum(r.total_area_mm2 for r in resources)
    _nonnegative("mandatory area", mandatory)
    die = HarvestingDie(name, node, mandatory, *resources)
    if not math.isclose(die.area, total, rel_tol=1e-12):
        raise ValueError("Area partition does not conserve physical die area")
    return die


def layout_report(die):
    return {
        "total_area_mm2": die.area,
        "mandatory_area_mm2": die.mandatory_area_mm2,
        "resources": {
            r.name: {
                "count": r.count,
                "total_area_mm2": r.total_area_mm2,
                "area_per_block_mm2": r.block_area,
                "units_per_block": r.units_per_block,
            }
            for r in die.resources
        },
    }
