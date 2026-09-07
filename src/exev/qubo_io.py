"""Portable, dependency-free serialization for ExEv QUBO models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from .qubo import QUBOModel

QUBO_FORMAT = "exev-qubo"
QUBO_FORMAT_VERSION = 1


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def qubo_to_dict(model: QUBOModel) -> dict[str, Any]:
    """Return a name-based JSON-compatible representation of a binary QUBO."""
    return {
        "format": QUBO_FORMAT,
        "format_version": QUBO_FORMAT_VERSION,
        "vartype": "BINARY",
        "variables": list(model.variable_names),
        "linear": [
            {
                "variable": model.variable_names[variable],
                "bias": coefficient,
            }
            for variable, coefficient in sorted(model.linear.items())
        ],
        "quadratic": [
            {
                "first": model.variable_names[first],
                "second": model.variable_names[second],
                "bias": coefficient,
            }
            for (first, second), coefficient in sorted(model.quadratic.items())
        ],
        "constant": model.constant,
    }


def qubo_from_dict(payload: Mapping[str, Any]) -> QUBOModel:
    """Validate and rebuild a QUBO serialized by :func:`qubo_to_dict`."""
    if not isinstance(payload, Mapping):
        raise ValueError("QUBO document must be a JSON object")
    if payload.get("format") != QUBO_FORMAT:
        raise ValueError(f"QUBO format must be {QUBO_FORMAT!r}")
    if payload.get("format_version") != QUBO_FORMAT_VERSION:
        raise ValueError(f"unsupported QUBO format version: {payload.get('format_version')!r}")
    if payload.get("vartype") != "BINARY":
        raise ValueError("QUBO vartype must be BINARY")

    variables = payload.get("variables")
    if (
        not isinstance(variables, Sequence)
        or isinstance(variables, (str, bytes))
        or any(not isinstance(name, str) or not name for name in variables)
    ):
        raise ValueError("variables must be a list of non-empty strings")
    if len(set(variables)) != len(variables):
        raise ValueError("QUBO variable names must be unique")

    model = QUBOModel()
    indices = {name: model.add_variable(name) for name in variables}
    model.constant = _finite_number(payload.get("constant"), "constant")

    linear = payload.get("linear")
    if not isinstance(linear, list):
        raise ValueError("linear must be a list")
    seen_linear: set[str] = set()
    for term in linear:
        if not isinstance(term, Mapping):
            raise ValueError("linear terms must be objects")
        variable = term.get("variable")
        if variable not in indices:
            raise ValueError(f"linear term references unknown variable: {variable!r}")
        if variable in seen_linear:
            raise ValueError(f"duplicate linear term for variable: {variable!r}")
        seen_linear.add(variable)
        model.add_linear(indices[variable], _finite_number(term.get("bias"), "linear bias"))

    quadratic = payload.get("quadratic")
    if not isinstance(quadratic, list):
        raise ValueError("quadratic must be a list")
    seen_quadratic: set[tuple[str, str]] = set()
    for term in quadratic:
        if not isinstance(term, Mapping):
            raise ValueError("quadratic terms must be objects")
        first, second = term.get("first"), term.get("second")
        if first not in indices or second not in indices:
            raise ValueError(
                f"quadratic term references unknown variables: {first!r}, {second!r}"
            )
        if first == second:
            raise ValueError("quadratic terms must reference two different variables")
        key = tuple(sorted((first, second)))
        if key in seen_quadratic:
            raise ValueError(f"duplicate quadratic term: {key!r}")
        seen_quadratic.add(key)
        model.add_quadratic(
            indices[first],
            indices[second],
            _finite_number(term.get("bias"), "quadratic bias"),
        )
    return model


def dumps_qubo(model: QUBOModel, *, indent: int | None = 2) -> str:
    return json.dumps(
        qubo_to_dict(model),
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    ) + ("\n" if indent is not None else "")


def loads_qubo(document: str) -> QUBOModel:
    try:
        payload = json.loads(document)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid QUBO JSON: {exc.msg}") from exc
    return qubo_from_dict(payload)


def write_qubo(model: QUBOModel, path: Path) -> Path:
    """Atomically write a QUBO document and return its absolute path."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(dumps_qubo(model))
    os.replace(temporary, destination)
    return destination


def read_qubo(path: Path) -> QUBOModel:
    return loads_qubo(path.read_text(encoding="utf-8"))


def qubo_fingerprint(model: QUBOModel) -> str:
    """Stable SHA-256 identity for the mathematical QUBO."""
    canonical = dumps_qubo(model, indent=None)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
