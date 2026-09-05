"""POT transport and full log-domain KL-unbalanced Sinkhorn."""

from __future__ import annotations

import warnings

import numpy as np
import ot
from scipy.special import logsumexp


class OptimalTransportError(RuntimeError):
    """Raised when an OT solver returns an unusable transport plan."""


class OptimalTransportConvergenceWarning(UserWarning):
    """Warn that a finite legacy plan did not meet the requested tolerance."""


_POT_FAILURE_WARNING_FRAGMENTS = (
    "did not converge",
    "numerical error",
    "numerical errors",
)


def _is_solver_failure_warning(warning: warnings.WarningMessage) -> bool:
    message = str(warning.message).lower()
    return issubclass(warning.category, RuntimeWarning) or any(
        fragment in message for fragment in _POT_FAILURE_WARNING_FRAGMENTS
    )


def _is_torch_tensor(value) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _to_numpy(value) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)) and any(_is_torch_tensor(item) for item in value):
        return np.asarray([float(item.detach().cpu().item()) if _is_torch_tensor(item) else item for item in value])
    return np.asarray(value)


def _backend_metadata(value) -> tuple[str, str, str]:
    if _is_torch_tensor(value):
        return ("torch", str(value.device), str(value.dtype).removeprefix("torch."))
    array = np.asarray(value)
    return ("numpy", "cpu", str(array.dtype))


def _stable_relative_change(current_log, previous_log) -> float:
    """POT's relative scaling change, without exponentiating large scalings."""
    if _is_torch_tensor(current_log):
        import torch

        scale = torch.maximum(
            torch.maximum(current_log.max(), previous_log.max()),
            current_log.new_tensor(0.0),
        )
        return float(torch.max(torch.abs(
            torch.exp(current_log - scale) - torch.exp(previous_log - scale)
        )).item())
    scale = max(float(current_log.max()), float(previous_log.max()), 0.0)
    return float(np.max(np.abs(
        np.exp(current_log - scale) - np.exp(previous_log - scale)
    )))


def _sinkhorn_log_unbalanced(M, a, b, *, reg, reg_m, numItermax, stopThr):
    """Generalized Sinkhorn for KL(P, a outer b) and KL marginal penalties.

    The kernel and scalings remain logarithmic until the final plan is formed.
    Histograms are normalized by the caller, as for the existing POT methods.
    """
    if _is_torch_tensor(M):
        import torch

        log, exp, zeros_like = torch.log, torch.exp, torch.zeros_like
        reduce_logsumexp = lambda value, axis: torch.logsumexp(value, dim=axis)
        all_finite = lambda value: bool(torch.isfinite(value).all().item())
    else:
        log, exp, zeros_like = np.log, np.exp, np.zeros_like
        reduce_logsumexp = logsumexp
        all_finite = lambda value: bool(np.isfinite(value).all())
    if float(a.min()) <= 0.0 or float(b.min()) <= 0.0:
        raise ValueError("sinkhorn_log requires strictly positive histogram entries.")
    log_a, log_b = log(a), log(b)
    log_kernel = -M / float(reg) + log_a[:, None] + log_b[None, :]
    exponent = float(reg_m) / (float(reg_m) + float(reg))
    log_u, log_v = zeros_like(a), zeros_like(b)
    errors = []
    for _ in range(int(numItermax)):
        previous_u, previous_v = log_u, log_v
        log_u = exponent * (
            log_a - reduce_logsumexp(log_kernel + log_v[None, :], axis=1)
        )
        log_v = exponent * (
            log_b - reduce_logsumexp(log_kernel + log_u[:, None], axis=0)
        )
        if not all_finite(log_u) or not all_finite(log_v):
            raise OptimalTransportError("Log-domain OT produced non-finite scaling potentials.")
        error = 0.5 * (
            _stable_relative_change(log_u, previous_u)
            + _stable_relative_change(log_v, previous_v)
        )
        errors.append(error)
        if error < float(stopThr):
            break
    plan = exp(log_u[:, None] + log_kernel + log_v[None, :])
    return plan, {"err": errors, "niter": len(errors)}


def sinkhorn_transport(
    M: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.05,
    numItermax: int = 1000,
    stopThr: float = 1e-5,
    unbalanced: bool = False,
    reg_m: float = 5.0,
    method: str = "sinkhorn",
    nonconvergence: str = "raise",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | bool | None]]:
    """Compute entropic-regularized OT via POT and verify solver status.

    ``nonconvergence='warn'`` makes a finite, nonconverged result explicit
    while retaining its numerical output. New callers should use the strict
    default.
    """
    if nonconvergence not in {"raise", "warn"}:
        raise ValueError("nonconvergence must be 'raise' or 'warn'.")
    expected_shape = tuple(M.shape)
    backend, device, dtype = _backend_metadata(M)
    a = a / a.sum()
    b = b / b.sum()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if unbalanced and method == "sinkhorn_log":
            plan, solver_log = _sinkhorn_log_unbalanced(
                M, a, b, reg=reg, reg_m=reg_m,
                numItermax=numItermax, stopThr=stopThr,
            )
        elif unbalanced:
            plan, solver_log = ot.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                method=method,
                numItermax=numItermax,
                stopThr=stopThr,
                log=True,
            )
        else:
            solver_result = ot.sinkhorn(
                a,
                b,
                M,
                reg,
                method=method,
                numItermax=numItermax,
                stopThr=stopThr,
                log=True,
            )
            if isinstance(solver_result, tuple) and len(solver_result) == 2:
                plan, solver_log = solver_result
            else:
                plan, solver_log = solver_result, {}

    solver_failures = [
        warning
        for warning in caught_warnings
        if _is_solver_failure_warning(warning)
    ]
    if solver_failures:
        details = "; ".join(str(warning.message) for warning in solver_failures)
        raise OptimalTransportError(f"OT solver reported failure: {details}")
    for warning in caught_warnings:
        warnings.warn_explicit(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
        )

    if not hasattr(solver_log, "get"):
        solver_log = {}
    error_history = _to_numpy(solver_log.get("err", [])).astype(np.float64, copy=False).reshape(-1)
    final_error = float(error_history[-1]) if error_history.size else None
    logged_iterations = solver_log.get("niter")
    iterations = (
        int(logged_iterations)
        if logged_iterations is not None
        else int(error_history.size)
    )
    converged = bool(
        final_error is not None
        and np.isfinite(final_error)
        and final_error < float(stopThr)
    )
    if not converged:
        solver_kind = "Unbalanced" if unbalanced else "Balanced"
        message = (
            f"{solver_kind} OT did not converge to the requested tolerance: "
            f"iterations={iterations}/{int(numItermax)}, "
            f"final_error={final_error!r}, stopThr={float(stopThr):g}."
        )
        if nonconvergence == "raise":
            raise OptimalTransportError(message)
        warnings.warn(message, OptimalTransportConvergenceWarning, stacklevel=2)

    plan = _to_numpy(plan).astype(np.float32, copy=False)
    if plan.shape != expected_shape:
        raise OptimalTransportError(
            f"OT solver returned shape {plan.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(plan)):
        raise OptimalTransportError("OT solver returned non-finite transport mass.")
    if np.any(plan < -1e-7):
        raise OptimalTransportError("OT solver returned negative transport mass.")
    if float(plan.sum(dtype=np.float64)) <= 0.0:
        raise OptimalTransportError("OT solver returned a massless transport plan.")
    if return_diagnostics:
        return plan, {
            "converged": converged,
            "iterations": iterations,
            "final_error": final_error,
            "stop_threshold": float(stopThr),
            "max_iterations": int(numItermax),
            "unbalanced": bool(unbalanced),
            "solver_method": str(method),
            "transport_backend": backend,
            "transport_device": device,
            "transport_dtype": dtype,
        }
    return plan
