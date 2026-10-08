# Monolithic die harvesting extension

This extension targets one physical die design sold as a nested high/mid/low
product family. Original `Chip`, `Module`, and packaging formulas are unchanged.
`spec.py` now locates `parameter.ini` relative to the checkout instead of cwd.
The baseline is upstream commit `42dd5de`. All shipped SKUs use the same full die
area and, in this first implementation, one shared organic-substrate package.

## Run

Use the existing `llmcompass_ae` environment (Python 3.9, SciPy 1.13.1 tested):

```bash
cd /home/sly/LLM/LLMCompassPlus/cost_model/chiplet-actuary
python -B -m chiplet_actuary.harvesting_cli --config examples/harvesting_5nm.json --output examples/harvesting_5nm_result.json
python -B -m unittest discover -s tests -v
```

Additional dependency: SciPy (for quadrature and stable binomial tails); already
available on GPU-3090. The `chiplet_actuary.harvesting` Python API provides
`Resource`, `Tier`, `HarvestingDie`, and `family_cost`.

The example is a **demonstration**, not a publication configuration. It uses a
Thor-derived 32-SM/16-core/64-MB-L2/512-bit geometry with upstream 5nm cost parameters.
Choosing `node: "5"` selects upstream wafer cost, defect density and NRE coefficients;
yield clustering uses the upstream shared manufacturing parameter. Geometry is
unchanged: selecting a cost node does not perform geometric process scaling.
The upstream 5nm parameters are not Thor-specific cost or defect-density measurements.
The 16-MB CPU L3, 32 L2 slices, 16 L3 slices, and 100,000 sales per tier in the
example are explicitly illustrative assumptions, not measured Thor properties.

## Physical defect partition

* GPU harvesting unit is fixed at **2 SM per TPC**, including SM private storage
  and the corresponding TPC-only extra area.
* CPU harvesting unit is fixed at **1 core**, including its private cache.
* GPU L2 and CPU shared L3 use equal-capacity, equal-area slices, with independently
  configurable slice counts. Local slice circuitry fails with its slice.
* GPU command front, shared GPC overhead, shared L2 controller, non-GPU/CPU IP and
  all unidentified/physical overhead are mandatory and must survive.
* Shared CPU L3 control is not independently identified in the area model; it is
  assumed included in mandatory system overhead. Slice-local circuitry is included
  in the measured L3 region.

`harvesting_adapter.from_soc_area` maps the existing area result into these four
harvestable pools plus mandatory area. The sum is checked against the complete
physical die. It never applies another utilization factor. Configurations with
`valid: false` from the area model are rejected. The CLI supports either `soc`
parameters plus `cache_slices`, or a manually supplied `layout` (not both).
The manual layout consists of `mandatory_area_mm2` and four entries named
`gpu_tpcs`, `cpu_cores`, `gpu_l2_slices`, `cpu_l3_slices`; each entry has `count`,
`total_area_mm2`, and `units_per_block` (2, 1, MB/slice, MB/slice respectively).

TPC/GPC separation uses the existing Thor area calibration; GPC shared circuitry
is conservatively mandatory. There is no topology constraint beyond resource
counts: healthy units can be selected arbitrarily, caches can be reconfigured,
and reconfiguration is assumed not to incur an additional performance penalty.
Timing binning, parametric faults, SRAM bit repair and manufacturing redundancy
are not modeled. These are architecture assumptions, not claims about NVIDIA's
actual harvesting implementation.

## Yield model

Let D be the selected defect density converted from defects/cm² to defects/mm²,
c the upstream `critical_level`, A0 mandatory area and ai the area of one unit in
resource pool i. Introduce a die-wide random intensity X ~ Gamma(c, rate=c).
Conditional on X=x:

```
P(mandatory survives | x) = exp(-D*x*A0)
P(a unit in pool i survives | x) = exp(-D*x*ai)
Gi | x ~ Binomial(ni, exp(-D*x*ai))
```

The four pools are independent conditional on X, but NOT independent marginally.
Integrate the mandatory survival probability times the four binomial tails over
X to obtain the probability of satisfying each tier. Quadrature reports failure
instead of silently accepting poor convergence. Requiring every unit healthy gives:

```
P(all good) = (1 + D*A_die/c)^(-c)
```

This exactly matches the upstream all-good negative-binomial yield. The common
Gamma intensity is one explicit extension consistent with that scalar model;
the upstream scalar yield does not uniquely determine the spatial fault model.
We do not independently apply the whole-die NB formula to every block and multiply.

Each tier declares positive minimum `gpu_sms`, `cpu_cores`, `gpu_l2_mb` and
`cpu_l3_mb`. GPU SM thresholds must be even. Cache thresholds round UP to complete
slices, and effective block requirements are included in output. All four minima
must hold simultaneously. Tiers must be listed highest first and componentwise
nested. Eligible yields are cumulative; adjacent differences produce mutually
exclusive highest-qualified bins. The remainder is scrap. Disabling harvesting
accepts only fully good dies, which can still be intentionally sold at lower tiers.

## Sales, wafer planning and manufacturing cost

Input demands are numbers of successfully packaged products. For the shared OS
package, upstream bonding yield y implies pre-package demand q_t/y per SKU.
Let p_t denote mutually exclusive bin probabilities, highest to lowest. The
minimum expected fabricated die count is:

```
N_required = max_k [sum(t<=k, q_t/y) / sum(t<=k, p_t)]
```

These cumulative constraints reserve enough high-qualified dies for high-tier
demand. Allocate each SKU's pre-package demand from its own bin first, then from
remaining higher bins. Lower bins can never satisfy higher-tier demand. Allocation
is reported as a source-bin by destination-SKU matrix, along with surplus and scrap.
Unsold good dies receive **zero** salvage/sales credit; the workload mix matters.

Use upstream gross dies per wafer and the full physical area to convert N_required
to wafers. `whole_wafers: true` rounds up wafer count, while false permits fractional
wafers for asymptotic comparisons. Both are expected-value models, not stochastic
service-level guarantees. Wafer cost covers every fabricated die, including scrap
and surplus. Bumping and raw package costs apply to pre-package demand; packaging
failures therefore consume additional graded dies and package inputs. Do not also
multiply this result by a separate die-yield loss factor.

Manufacturing total is wafer cost + bumping + raw package costs for these attempts.
With harvesting disabled and fractional wafers, cost per shipped unit reproduces
the original one-die `OS.cost_total_system()` result, including bonding losses.
The physical die/module design NRE and shared package NRE are charged once across
the family, using upstream formulas. Their sum is independent of which bin a die
enters. No separate NRE is charged for creating the three sales labels.

The outputs include family manufacturing total, NRE, total cost, average cost per
shipped unit, and enabled/disabled harvesting comparisons. Per-SKU selling prices
or arbitrary accounting allocations are intentionally not inferred. Inspection
and binning overhead, board/DRAM costs, inventory carry and reticle constraints
are outside this extension. Package variants and multi-die product-family costing
remain separate work; this extension is the harvested monolithic baseline.

## Verification

Tests compare integrated probabilities with an independent finite
inclusion-exclusion enumeration on a small example, recover original all-good
yield and unharvested manufacturing cost, exercise cache-only recovery, require
GPU and CPU simultaneously, enforce downgrade-only allocation, conserve
fabricated/scrap/unused/consumed die counts, charge NRE once, and verify that
high-only demand receives no fictitious credit from unsold lower bins.
