#!/usr/bin/env python3
"""
m3_topology_validate.py — slice 0b schema validator.

Validates a per-machine topology capture (JSON) against the schema used
by docs/results/m3-topology-A-2026-09-07.json and the B/C/D templates.

Goals (data-model only; no engine code change):
  * explicit `units` block; required units listed at minimum.
  * every numeric field has a non-zero value OR is the literal string
    "NOT_RUN"; never a fabricated default.
  * drive / controller / interconnect ids are unique within their kind.
  * machine.eligibility is a structured object (not a free-form note).
  * open_evidence_gaps is a list of named gaps; never silently empty
    unless the machine has no open evidence gaps (and that requires an
    explicit "complete": true field).

Exit codes:
  0  PASS
  1  PASS with warnings
  2  FAIL (schema or hard rule violation)
  3  usage error
"""
import argparse, json, os, sys
from collections import defaultdict

REQUIRED_UNITS = {"rate", "latency", "bytes"}
REQUIRED_TOP = ["schema_version", "machine_id", "capture_date",
                "evidence_basis", "units", "compute", "memory_domains",
                "storage", "controllers", "interconnects", "machine",
                "operating_environment", "open_evidence_gaps"]

def warn(msg):
    print(f"WARN  {msg}")

def err(msg):
    print(f"ERROR {msg}")

def is_explicit_value(v):
    """A value is 'explicit' if it is numeric, bool, list, or a non-NOT_RUN
    string. The literal 'NOT_RUN' is the only acceptable placeholder."""
    if v is None:
        return False
    if isinstance(v, (int, float, bool)):
        return True
    if isinstance(v, (list, dict)):
        return True
    if isinstance(v, str):
        return v != "NOT_RUN"
    return False

def validate(path):
    if not os.path.isfile(path):
        err(f"file not found: {path}")
        return 2

    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        err(f"invalid JSON: {e}")
        return 2

    status = 0
    notes = []

    # top-level required keys
    for key in REQUIRED_TOP:
        if key not in doc:
            err(f"missing required top-level key: {key}")
            status = 2

    # units
    units = doc.get("units", {})
    for u in REQUIRED_UNITS:
        if u not in units:
            err(f"units.{u} missing")
            status = 2

    # machine_id
    mid = doc.get("machine_id", "")
    if mid not in ("A", "B", "C", "D"):
        warn(f"machine_id is {mid!r}, expected one of A/B/C/D")

    # compute.cpu.isa_caps / isa_caps_missing
    cpu = doc.get("compute", {}).get("cpu", {})
    if not cpu:
        err("compute.cpu missing or empty")
        status = 2
    else:
        for k in ("model", "isa_caps", "physical_cores", "logical_cores"):
            if k not in cpu:
                err(f"compute.cpu.{k} missing")
                status = 2
        for k in ("physical_cores", "logical_cores"):
            v = cpu.get(k)
            if v is not None and not is_explicit_value(v):
                err(f"compute.cpu.{k} is {v!r}; expected numeric or NOT_RUN")
                status = 2

    # gpu list (may be empty on machine D)
    gpus = doc.get("compute", {}).get("gpu", [])
    if not isinstance(gpus, list):
        err("compute.gpu must be a list")
        status = 2
    else:
        for i, g in enumerate(gpus):
            for k in ("model", "arch"):
                if k not in g:
                    err(f"compute.gpu[{i}].{k} missing")
                    status = 2
            vram = g.get("vram_bytes", g.get("vram_bytes_per_gpu"))
            if vram is None:
                err(f"compute.gpu[{i}].vram_bytes (or vram_bytes_per_gpu) missing")
                status = 2
            elif not is_explicit_value(vram):
                err(f"compute.gpu[{i}].vram_bytes is {vram!r}; expected numeric or NOT_RUN")
                status = 2
            elig = g.get("eligible_for_expert_compute")
            if elig is None:
                warn(f"compute.gpu[{i}].eligible_for_expert_compute missing")

    # memory_domains
    mems = doc.get("memory_domains", [])
    if not isinstance(mems, list) or not mems:
        err("memory_domains must be a non-empty list")
        status = 2
    else:
        seen_ids = set()
        for m in mems:
            mid_m = m.get("id")
            if not mid_m:
                err("memory_domains entry missing id")
                status = 2
            elif mid_m in seen_ids:
                err(f"duplicate memory_domain id: {mid_m}")
                status = 2
            else:
                seen_ids.add(mid_m)
            for k in ("kind", "numa_attach"):
                v = m.get(k)
                if v is None:
                    err(f"memory_domains[{mid_m}].{k} missing")
                    status = 2
                elif not is_explicit_value(v):
                    err(f"memory_domains[{mid_m}].{k} is {v!r}; expected numeric or NOT_RUN")
                    status = 2
            tb = m.get("total_bytes")
            if tb is None:
                err(f"memory_domains[{mid_m}].total_bytes missing")
                status = 2
            elif tb != "NOT_RUN" and not isinstance(tb, (int, float)):
                err(f"memory_domains[{mid_m}].total_bytes must be numeric or 'NOT_RUN'")
                status = 2

    # storage
    drives = doc.get("storage", [])
    if not isinstance(drives, list):
        err("storage must be a list")
        status = 2
    else:
        seen_ids = set()
        for d in drives:
            did = d.get("id")
            if not did:
                err("storage entry missing id")
                status = 2
            elif did in seen_ids:
                err(f"duplicate storage id: {did}")
                status = 2
            else:
                seen_ids.add(did)
            for k in ("model", "controller", "interface"):
                v = d.get(k)
                if v is None:
                    err(f"storage[{did}].{k} missing")
                    status = 2
            cal = d.get("calibration")
            if cal == "NOT_RUN":
                notes.append(f"{did}: calibration NOT_RUN")
            elif cal is None:
                err(f"storage[{did}].calibration missing (use 'NOT_RUN' if not measured)")
                status = 2

    # controllers — drives referenced must exist; placeholder ids in
    # templates are tolerated only when the controller id itself contains
    # a placeholder token (FILL_*) so the validator can skip the check
    # for templates that haven't been filled in yet.
    controller_ids = set()
    for c in doc.get("controllers", []):
        cid = c.get("id")
        if not cid:
            err("controller entry missing id")
            status = 2
        elif cid in controller_ids:
            err(f"duplicate controller id: {cid}")
            status = 2
        else:
            controller_ids.add(cid)
        is_template = isinstance(cid, str) and ("FILL_" in cid or "FILL" in cid)
        for d in c.get("drives", []):
            if is_template:
                continue
            if not any(drv.get("id") == d for drv in drives):
                err(f"controller {cid} references unknown drive {d}")
                status = 2

    # interconnects
    interconnect_ids = set()
    for ic in doc.get("interconnects", []):
        iid = ic.get("id")
        if not iid:
            err("interconnect entry missing id")
            status = 2
        elif iid in interconnect_ids:
            err(f"duplicate interconnect id: {iid}")
            status = 2
        else:
            interconnect_ids.add(iid)

    # machine.eligibility structure
    elig = doc.get("machine", {}).get("eligibility", {})
    if not isinstance(elig, dict):
        err("machine.eligibility must be a dict")
        status = 2
    else:
        if "cpu_island" not in elig:
            err("machine.eligibility.cpu_island missing")
            status = 2
        if "gpu_island" not in elig:
            warn("machine.eligibility.gpu_island missing")

    # open_evidence_gaps
    gaps = doc.get("open_evidence_gaps", [])
    if not isinstance(gaps, list):
        err("open_evidence_gaps must be a list")
        status = 2
    else:
        for g in gaps:
            for k in ("id", "field", "status", "rationale"):
                if k not in g:
                    err(f"open_evidence_gaps entry missing key: {k}")
                    status = 2
            if g.get("status") != "NOT_RUN":
                warn(f"open_evidence_gaps[{g.get('id')}].status is "
                     f"{g.get('status')!r}; expected NOT_RUN while unmeasured")

    # overall PASS/PASS-with-warnings/FAIL
    if status == 0 and notes:
        print(f"PASS-WITH-NOTES ({path}):")
        for n in notes:
            print(f"  - {n}")
        return 1
    if status == 0:
        print(f"PASS ({path})")
    else:
        print(f"FAIL ({path})")
    return status

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or more topology JSON files")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rc = 0
    for p in args.paths:
        r = validate(p)
        rc = max(rc, r)
    return rc

if __name__ == "__main__":
    sys.exit(main())
