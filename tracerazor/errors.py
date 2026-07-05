"""Typed exceptions for the TraceRazor Python API."""

from __future__ import annotations


class TraceRazorError(Exception):
    """Base class for TraceRazor Python API errors."""


class BinaryNotFoundError(TraceRazorError, FileNotFoundError):
    """Raised when the Rust auditor binary cannot be resolved."""


class AuditError(TraceRazorError, RuntimeError):
    """Raised when an audit command or server request fails."""


class BelowMinStepsError(AuditError):
    """Raised when a trace is valid but below the configured audit step floor."""


class VerificationError(TraceRazorError, RuntimeError):
    """Raised when verification cannot complete or detects tampering."""
