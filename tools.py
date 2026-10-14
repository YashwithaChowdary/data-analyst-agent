"""Reusable analysis tools for the data analyst agent.

Every tool takes the current DataFrame plus plain arguments and returns a
ToolResult. A tool never raises on bad input: it returns ok=False with a
message a person (or a model) can act on.

- table: what to show the user (a DataFrame), or None
- frame: the DataFrame later steps should work on, or None to keep the current one
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

AGGS = ("sum", "mean", "median", "min", "max", "count")
OPS = ("==", "!=", ">", ">=", "<", "<=", "contains")


@dataclass
class ToolResult:
    ok: bool
    table: Optional[pd.DataFrame] = None
    frame: Optional[pd.DataFrame] = None
    note: str = ""
    error: str = ""


def _fail(message):
    return ToolResult(ok=False, error=message)


def _missing_column(df, column):
    if column not in df.columns:
        return (f"Unknown column '{column}'. Available columns: "
                f"{', '.join(map(str, df.columns))}")
    return None


def load_csv(path):
    """Load a CSV file. Returns (DataFrame, error message)."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except pd.errors.EmptyDataError:
        return None, f"File is empty: {path}"
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"Could not read {path} as CSV: {exc}"
    if df.empty:
        return None, f"File has headers but no rows: {path}"
    return df, ""


def describe(df):
    """Row count, column types, missing values and numeric summary."""
    info = pd.DataFrame({"type": df.dtypes.astype(str),
                         "missing": df.isna().sum()})
    numeric = df.select_dtypes("number")
    note = f"{len(df)} rows, {len(df.columns)} columns"
    if not numeric.empty:
        info = info.join(numeric.agg(["mean", "min", "max"]).T.round(2))
    return ToolResult(True, table=info.reset_index(names="column"), note=note)


def filter_rows(df, column, op, value):
    """Keep rows where column <op> value. Later steps see the filtered rows."""
    problem = _missing_column(df, column)
    if problem:
        return _fail(problem)
    if op not in OPS:
        return _fail(f"Unknown operator '{op}'. Use one of: {', '.join(OPS)}")
    series = df[column]
    if op == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    else:
        if pd.api.types.is_numeric_dtype(series):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return _fail(f"Column '{column}' is numeric, but '{value}' is not a number")
        elif op not in ("==", "!="):
            return _fail(f"Operator '{op}' needs a numeric column; '{column}' is text")
        else:
            value = str(value)
        compare = {"==": series.eq, "!=": series.ne, ">": series.gt,
                   ">=": series.ge, "<": series.lt, "<=": series.le}[op]
        mask = compare(value)
    kept = df[mask.fillna(False)]
    return ToolResult(True, table=kept.head(10), frame=kept,
                      note=f"{len(kept)} of {len(df)} rows match {column} {op} {value}")


def group_summary(df, by, column, agg="sum"):
    """Aggregate `column` for each value of `by`, largest first."""
    for name in (by, column):
        problem = _missing_column(df, name)
        if problem:
            return _fail(problem)
    if agg not in AGGS:
        return _fail(f"Unknown aggregation '{agg}'. Use one of: {', '.join(AGGS)}")
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[column]):
        return _fail(f"Cannot take the {agg} of '{column}': it is not numeric")
    out = df.groupby(by)[column].agg(agg).sort_values(ascending=False)
    out = out.round(2).reset_index()
    out.columns = [by, f"{agg}_of_{column}"]
    return ToolResult(True, table=out, note=f"{agg} of {column} by {by}")


def top_n(df, column, n=5, ascending=False):
    """The n rows with the largest (or smallest) values in `column`."""
    problem = _missing_column(df, column)
    if problem:
        return _fail(problem)
    if not pd.api.types.is_numeric_dtype(df[column]):
        return _fail(f"Cannot rank by '{column}': it is not numeric")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return _fail(f"n must be a whole number, got '{n}'")
    if n < 1:
        return _fail("n must be at least 1")
    out = df.sort_values(column, ascending=bool(ascending)).head(n)
    return ToolResult(True, table=out, note=f"top {len(out)} by {column}")


def value_counts(df, column, n=10):
    """How often each value of `column` occurs."""
    problem = _missing_column(df, column)
    if problem:
        return _fail(problem)
    counts = df[column].value_counts().head(int(n)).reset_index()
    counts.columns = [column, "count"]
    return ToolResult(True, table=counts, note=f"counts of {column}")


# name -> (function, allowed argument names)
TOOLS = {
    "describe": (describe, ()),
    "filter_rows": (filter_rows, ("column", "op", "value")),
    "group_summary": (group_summary, ("by", "column", "agg")),
    "top_n": (top_n, ("column", "n", "ascending")),
    "value_counts": (value_counts, ("column", "n")),
}
