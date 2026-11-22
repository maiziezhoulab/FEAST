"""Run the registered PASTE2 fixed-plate jobs from the staged rotations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

INPUTS = Path(__file__).resolve().parents[2] / "Zenodo/inputs/02_alignment"


def run_job(args, rotation, public_seed):
    alteration = rotation.alteration
    angle = float(rotation.angle_degrees)
    output = args.output_dir / alteration / f"angle_{angle:g}"
    moving = args.rotation_dir / rotation.output_path
    worker = Path(__file__).parent / "methods/run_paste2.py"
    configuration = f"study02-paste2-{alteration}-{angle:g}"
    command = [
        str(args.python), str(worker), "--reference", str(args.reference), "--moving", str(moving),
        "--output-dir", str(output), "--seed", str(public_seed), "--configuration-id", configuration,
        "--overlap-fraction", repr(float(rotation.retained_fraction)), "--config", str(args.config),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f"angle_{angle:g}.runner.log"
    started = time.monotonic()
    print(f"Running PASTE2 {alteration}/{angle:g}°", flush=True)
    with log_path.open("w", buffering=1) as log:
        log.write(shlex.join(command) + "\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    status = "ok" if completed.returncode == 0 else "failed"
    artifacts = [{"path": path.name} for path in sorted(output.glob("*")) if path.is_file()]
    return {
        "job_id": f"paste2__{alteration}__angle_{angle:g}",
        "configuration_id": configuration, "method": "paste2", "alteration": alteration,
        "angle_degrees": angle, "retained_fraction": float(rotation.retained_fraction),
        "feast_version": rotation.feast_version, "public_seed": public_seed,
        "reference": os.path.relpath(args.reference, args.output_dir),
        "moving": os.path.relpath(moving, args.output_dir),
        "output_dir": output.relative_to(args.output_dir).as_posix(),
        "status": status, "return_code": completed.returncode,
        "method_python": str(args.python.absolute()), "artifacts": json.dumps(artifacts),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=INPUTS / "paste2_config.yaml")
    parser.add_argument("--rotation-config", type=Path, help="Main fixed-plate geometry configuration")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent independent CPU solver jobs")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    config = yaml.safe_load((args.rotation_config or args.config.with_name("config.yaml")).read_text())
    rotations = pd.read_csv(args.rotation_dir / "rotation_manifest.csv").sort_values(["alteration", "angle_degrees"])
    expected = {(alteration, float(angle)) for alteration in config["alterations"] for angle in config["angles"]}
    if len(rotations) != len(expected) or set(zip(rotations.alteration, rotations.angle_degrees)) != expected:
        raise RuntimeError("Rotation manifest does not cover the registered PASTE2 grid")
    public_seed = int(config["seed"])
    if not rotations.public_seed.eq(public_seed).all():
        raise RuntimeError("PASTE2 rotation inputs have a different simulation seed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(run_job, args, rotation, public_seed) for rotation in rotations.itertuples(index=False)]
        for job in as_completed(jobs):
            row = job.result()
            rows.append(row)
            pd.DataFrame(rows).sort_values(["alteration", "angle_degrees"]).to_csv(
                args.output_dir / "method_manifest.csv", index=False,
            )
            print(f"[{len(rows)}/{len(jobs)}] {row['job_id']}: {row['status']}", flush=True)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
