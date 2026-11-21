"""Shared public vocabulary for differential-expression comparisons."""

from __future__ import annotations

from typing import Literal

DETargetName = Literal["gt_half", "gt_all_cells"]
DEReferenceName = Literal["all_perturbed", "control"]

DE_TARGETS: tuple[DETargetName, ...] = ("gt_half", "gt_all_cells")
DE_REFERENCES: tuple[DEReferenceName, ...] = ("all_perturbed", "control")


def validate_de_target(target: str) -> None:
    """Reject a target outside the supported DE comparison vocabulary."""
    if target not in DE_TARGETS:
        raise ValueError(f"unsupported DE target {target!r}; available: {', '.join(DE_TARGETS)}")


def validate_de_reference(reference: str) -> None:
    """Reject a reference outside the supported DE comparison vocabulary."""
    if reference not in DE_REFERENCES:
        raise ValueError(f"unsupported DE reference {reference!r}; available: {', '.join(DE_REFERENCES)}")
