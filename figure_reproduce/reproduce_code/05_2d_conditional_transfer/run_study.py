"""Run the registered 45 Study 05 jobs, scoring and ten-draw comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from workflow import STUDY_DIR, declared_jobs, expected_job_path, job_id, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feast-source", type=Path, required=True, help="FEAST checkout or unpacked source archive")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--output-root", type=Path, help="Write a separate reproduction with the same deposited inputs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    source = args.feast_source.resolve()
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != args.source_revision:
            raise ValueError(f"FEAST checkout revision is {revision}")
    inputs = args.zenodo / "inputs" / STUDY_DIR.name
    results = args.output_root or args.zenodo / "results" / STUDY_DIR.name
    results.mkdir(parents=True, exist_ok=True)
    logs = results / "logs"
    logs.mkdir(exist_ok=True)
    config_path = STUDY_DIR / "config.yaml"
    config = load_config(config_path)
    (results / "run_config.yaml").write_text(config_path.read_text())
    executed_manifest = results / "executed_input_files.csv"
    if not executed_manifest.exists():
        executed_manifest.write_text((inputs / "input_files.csv").read_text())
    data_args = [
        "--config", os.path.relpath(config_path), "--manifest", str(inputs / "input_files.csv"),
        "--input-dir", str(inputs),
    ]
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), FEAST_SOURCE_REVISION=args.source_revision)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    identity = subprocess.check_output(
        [sys.executable, "-c", "import FEAST; print(FEAST.__version__); print(FEAST.__file__)"],
        env=environment, text=True,
    ).splitlines()
    if identity != [str(config["required_feast_version"]), str(source / "src/FEAST/__init__.py")]:
        raise ValueError(f"Unexpected FEAST import: {identity}")
    print(*identity, flush=True)

    def run(script, arguments, log):
        command = [sys.executable, os.path.relpath(STUDY_DIR / script), *data_args, *map(str, arguments)]
        with (logs / f"{log}.log").open("w") as handle:
            handle.write(json.dumps(command) + "\n")
            handle.flush()
            subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(log, flush=True)

    preflight = results / "preflight.json"
    if not preflight.exists():
        run("preflight.py", ["--report", preflight], "preflight")
    generation = results / "final"

    def run_job(job):
        directory = expected_job_path(generation, job)
        if directory.exists():
            record = json.loads((directory / "provenance.json").read_text())
            if record["runtime"]["source_revision"] != args.source_revision:
                raise ValueError(f"Different source revision in {directory}")
            return
        options = ["--mode", job.mode, "--dataset", job.dataset, "--output-dir", generation]
        if job.mode == "cross_slice":
            options += ["--source", job.source, "--target", job.target, "--assignment-randomness", job.assignment_randomness]
        else:
            options += ["--slice", job.source]
        run("run.py", options, job_id(config, job))

    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        list(workers.map(run_job, declared_jobs(config)))
    scores = results / "scores"
    if not scores.exists():
        run("score.py", ["--generation-dir", generation, "--output-dir", scores], "score")
    validation = results / "study05_complete_validation.json"
    if not validation.exists():
        run("validate.py", ["--all", "--output-dir", generation, "--scores-dir", scores, "--report", validation], "validate")
    comparison = results / "baselines/conditional_metric_comparison"
    if not comparison.exists():
        run("evaluate_resampling_metrics.py",
            ["--feast-root", generation, "--replicates", 10, "--output-dir", comparison], "resampling")


if __name__ == "__main__":
    main()
