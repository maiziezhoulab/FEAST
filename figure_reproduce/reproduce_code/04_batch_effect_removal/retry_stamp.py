"""Apply the registered STAMP retry to one documented 04A stress-test failure."""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import yaml

import run as workflow
from score import REPRODUCTION_ROOT, STAMP_RETRY_POLICY_PATH, validate_stamp_retry


def verify_failure(run, failure, mode, alpha, standard_patience, retry_patience, method_settings):
    if failure.parent != run / "failures" or not failure.name.startswith(f"STAMP__{mode}__alpha_{alpha:.2f}__"):
        raise ValueError("The failure directory must belong to this exact run and stress cell")
    error = (failure / "FAILED.txt").read_text()
    disposition = (failure / "DISPOSITION.txt").read_text().strip()
    if disposition != "exit=1; missing metadata.json" or not all(
        token in error for token in ("caux", "tensor([nan]", "cuda:0")
    ):
        raise ValueError("The recorded failure does not show the registered CUDA Pyro non-finiteness")
    log = run / "logs" / "STAMP" / mode / f"alpha_{alpha:.2f}.log"
    contents = log.read_text()
    invocation = shlex.split(contents.splitlines()[0])
    if invocation[invocation.index("--patience") + 1] != str(standard_patience):
        raise ValueError("The original attempt did not use the registered patience")
    maximum = method_settings["max_epochs"]
    minimum = method_settings["min_epochs"]
    pattern = rf"Epoch Loss:\s*([+\-\d.eE]+):.*?\b(\d+)/{maximum}\b"
    losses = {int(epoch): float(loss) for loss, epoch in re.findall(pattern, contents) if int(epoch) > minimum}
    if len(losses) < retry_patience + 2:
        raise ValueError("The original fit log lacks a finite plateau after the minimum epochs")
    failure_epoch = max(losses)
    losses.pop(failure_epoch)  # The failure printed the last loss again before finishing this epoch.
    best = min(losses.values())
    best_epoch = min(epoch for epoch, value in losses.items() if value == best)
    non_improving = sum(loss > best for epoch, loss in losses.items() if epoch > best_epoch)
    if not retry_patience <= non_improving < standard_patience:
        raise ValueError(f"The logged finite plateau has {non_improving} non-improving epochs")
    return {
        "standard_attempt_log": log.relative_to(REPRODUCTION_ROOT).as_posix(),
        "best_logged_loss": best,
        "best_logged_epoch_index": best_epoch,
        "failure_epoch_index": failure_epoch,
        "non_improving_epochs": non_improving,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=workflow.STUDY_ROOT / "config.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["shift_only", "diagonal_affine"], required=True)
    parser.add_argument("--failure-dir", type=Path, required=True)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    feast = workflow.feast_identity(config)
    run = args.run_dir.resolve()
    mode, alpha = args.mode, 1.5
    policy_path = REPRODUCTION_ROOT / STAMP_RETRY_POLICY_PATH
    policy = yaml.safe_load(policy_path.read_text())
    if {"mode": mode, "alpha": alpha} not in policy["eligible_conditions"]:
        raise ValueError("This condition is outside the registered retry")
    if policy["parent_configuration_id"] != config["configuration_id"]:
        raise ValueError("The retry policy belongs to a different study configuration")
    change = policy["retry_change"]
    method = config["methods"]["STAMP"]
    settings = method["kwargs"].copy()
    standard, retry = change["standard_value"], change["retry_value"]
    if settings["patience"] != standard:
        raise ValueError("The configured original STAMP patience differs from the policy")
    failure = args.failure_dir.resolve()
    evidence = verify_failure(run, failure, mode, alpha, standard, retry, settings)
    settings["patience"] = retry
    candidate = run / "methods" / "STAMP" / mode / "alpha_1.50"
    if candidate.exists():
        raise FileExistsError(candidate)
    python = workflow.resolve_path(config["_config_path"], method["python"])
    worker = workflow.resolve_path(config["_config_path"], method["script"])
    seed = int(config["public_seed"])
    candidate_id = f"{config['configuration_id']}__STAMP__{mode}__alpha_1.50"
    command = [
        str(python), str(worker),
        "--reference", str(run / f"simulations/{mode}/alpha_0.00.h5ad"),
        "--query", str(run / f"simulations/{mode}/alpha_1.50.h5ad"),
        "--output-dir", str(candidate),
        "--panel-file", str(run / "panel/panel.csv"),
        "--panel-provenance", str(run / "panel/provenance.json"),
        "--seed", str(seed), "--config-id", candidate_id,
        "--parent-config-id", config["configuration_id"],
        "--deformation-mode", mode, "--alpha", str(alpha), "--config-path", str(config["_config_path"]),
    ]
    for name, value in settings.items():
        command.extend([f"--{name.replace('_', '-')}", workflow.format_cli_value(value)])
    environment = {
        **os.environ, "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8",
        "FEAST_REPO": str(config["_feast_repo"]), "FEAST_VERSION": feast["version"],
    }
    subprocess.run([str(python), "-m", "pip", "check"], env=environment, check=True)
    environment["FEAST_REPRODUCE_PIP_CHECK"] = "ok"
    log_path = run / "logs" / "STAMP" / mode / "alpha_1.50_retry.log"
    print(f"Retry STAMP/{mode}/alpha_1.50 after {evidence['non_improving_epochs']} non-improving epochs", flush=True)
    with log_path.open("x", buffering=1) as log:
        log.write(shlex.join(command) + "\n")
        subprocess.run(command, cwd=workflow.STUDY_ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata_path = candidate / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["numerical_retry"] = {
        "policy_configuration_id": policy["configuration_id"],
        "policy_path": STAMP_RETRY_POLICY_PATH,
        "eligible_condition": {"mode": mode, "alpha": alpha},
        "changed_parameter": {"name": "patience", "standard_value": standard, "retry_value": retry},
        "unchanged_contract": policy["unchanged_contract"],
        "activation_reason": "A non-finite Pyro caux parameter followed a finite loss plateau before standard patience elapsed.",
        "original_failure_path": (failure / "FAILED.txt").relative_to(REPRODUCTION_ROOT).as_posix(),
        "original_disposition_path": (failure / "DISPOSITION.txt").relative_to(REPRODUCTION_ROOT).as_posix(),
        **evidence,
    }
    metadata["provenance"]["sources"].append({"path": str(policy_path)})
    validate_stamp_retry(metadata, candidate)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    valid, reason = workflow.candidate_is_valid(candidate, "STAMP", candidate_id, config)
    if not valid:
        raise ValueError(f"The retry candidate failed validation: {reason}")
    print(f"Verified {candidate}")


if __name__ == "__main__":
    main()
