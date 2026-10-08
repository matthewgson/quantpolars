"""Bridge to the compiled Rust expression plugins in ``quantpolars._internal``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent

# Anything that can stand in for a column: a name, an expression, or a constant.
ExprLike = Union[str, pl.Expr, float, int]
FrameLike = Union[pl.DataFrame, pl.LazyFrame]
OptionType = Union[str, bool, pl.Expr]


def to_expr(x: ExprLike) -> pl.Expr:
    """Column name -> pl.col, number -> pl.lit, expression -> itself."""
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    return pl.lit(float(x))


def keep_name(x: ExprLike, expr: pl.Expr) -> pl.Expr:
    """If the input was a column name, the output keeps that name."""
    return expr.alias(x) if isinstance(x, str) else expr


def option_flag(option_type: OptionType) -> pl.Expr:
    """'call'/'c' or 'put'/'p' (any case) are literals; any other string is a column
    name. Columns are parsed in Rust (Boolean, numeric +1/-1, or string flags)."""
    if isinstance(option_type, pl.Expr):
        return option_type
    if isinstance(option_type, bool):
        return pl.lit(option_type)
    if isinstance(option_type, str):
        token = option_type.strip().lower()
        if token in ("call", "c"):
            return pl.lit(True)
        if token in ("put", "p"):
            return pl.lit(False)
        return pl.col(option_type)
    raise TypeError(f"option_type must be 'call', 'put', a column name or an expression, got {option_type!r}")


def call(function_name: str, *args: pl.Expr, kwargs: Optional[Dict[str, Any]] = None) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        function_name=function_name,
        args=list(args),
        kwargs=kwargs,
        is_elementwise=True,
    )
