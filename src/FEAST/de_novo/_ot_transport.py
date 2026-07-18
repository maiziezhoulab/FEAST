"""Sinkhorn transport wrapper using POT (Python Optimal Transport)."""

from __future__ import annotations

import warnings

import numpy as np
import ot


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


def sinkhorn_transport(
    M: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.05,
    numItermax: int = 1000,
    stopThr: float = 1e-5,
    unbalanced: bool = False,
    reg_m: float = 5.0,
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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if unbalanced:
            plan, solver_log = ot.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
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
    error_history = np.asarray(
        solver_log.get("err", []),
        dtype=np.float64,
    ).reshape(-1)
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

    plan = np.asarray(plan, dtype=np.float32)
    if plan.shape != M.shape:
        raise OptimalTransportError(
            f"OT solver returned shape {plan.shape}; expected {M.shape}."
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
        }
    return plan
