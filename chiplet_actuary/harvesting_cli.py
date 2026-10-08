"""Run: python -m chiplet_actuary.harvesting_cli --config examples/harvesting_7nm.json"""

import argparse
import json
import sys
from pathlib import Path

from chiplet_actuary.harvesting import HarvestingDie, Resource, Tier, family_cost
from chiplet_actuary.harvesting_adapter import from_soc_area, layout_report


def make_die(config):
    name, node = config["name"], str(config["node"])
    if "layout" in config:
        if "soc" in config:
            raise ValueError("Specify either layout or soc, not both")
        layout = config["layout"]
        resources = [
            Resource(key, **layout[key])
            for key in ("gpu_tpcs", "cpu_cores", "gpu_l2_slices", "cpu_l3_slices")
        ]
        return HarvestingDie(name, node, layout["mandatory_area_mm2"], *resources), None
    # This extension lives in LLMCompassPlus/cost_model/chiplet-actuary.
    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace))
    from area_model.area_model import load_area_data
    from area_model.soc_model import calculate_soc_area

    area = calculate_soc_area(**config["soc"])
    calibration = load_area_data("Thor")
    tpc = calibration["tpc_extra"]
    tpc_area = (
        tpc["unit_count"]
        * tpc["area_per_unit"]
        * area["config"]["physical_sm_count"]
        / calibration["sm"]["unit_count"]
    )
    die = from_soc_area(
        area,
        name=name,
        node=node,
        gpu_tpc_extra_area_mm2=tpc_area,
        gpu_l2_shared_control_area_mm2=calibration["l2_cache"]["ctrl"],
        **config["cache_slices"],
    )
    return die, area


def run_config(config):
    die, area = make_die(config)
    tiers = [Tier(**tier) for tier in config["tiers"]]
    whole = config.get("whole_wafers", True)
    harvested = family_cost(die, tiers, whole_wafers=whole)
    baseline = family_cost(die, tiers, whole_wafers=whole, harvesting=False)
    result = {
        "scenario_label": config.get("scenario_label", "user-supplied scenario"),
        "layout": layout_report(die),
        "with_harvesting": harvested,
        "without_harvesting": baseline,
        "assumptions": [
            "Gamma-Poisson die-wide intensity; block failures conditionally independent.",
            "Any healthy TPC/core/cache slice is usable; routing, timing and repair implementation are not modeled.",
            "Common control, all other IP and all unidentified area overhead must be defect-free.",
            "One physical die design and one shared OS package design across all SKUs.",
            "Cost-node selection does not scale the input area to that technology.",
        ],
    }
    if area is not None:
        result["soc_area_result"] = area
    if harvested["feasible"] and baseline["feasible"]:
        result["manufacturing_saving_fraction"] = (
            1 - harvested["manufacturing_total"] / baseline["manufacturing_total"]
        )
        result["total_saving_fraction"] = (
            1 - harvested["family_total"] / baseline["family_total"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        with args.config.open(encoding="utf-8") as stream:
            result = run_config(json.load(stream))
        text = json.dumps(result, indent=2, allow_nan=False)
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        parser.error(str(exc))
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
