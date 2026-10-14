"""Plan, validate and run analysis steps.

A plan is a list of steps: {"tool": "group_summary", "args": {...}}.
It can be written by hand or proposed by a model (plan_with_llm). Either way
it is validated before anything runs, so a wrong column name or an invented
tool is caught up front with a clear message.

The model only ever sees column names and types, never the rows.
"""
import json
import re
from dataclasses import dataclass, field

from tools import TOOLS


@dataclass
class Report:
    question: str
    plan: list
    tables: list = field(default_factory=list)     # (note, DataFrame)
    errors: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors


def validate_plan(plan, columns):
    """Return a list of problems. An empty list means the plan can run."""
    if not isinstance(plan, list) or not plan:
        return ["The plan must be a non-empty list of steps"]
    problems = []
    for i, step in enumerate(plan, 1):
        if not isinstance(step, dict) or "tool" not in step:
            problems.append(f"Step {i}: each step needs a 'tool'")
            continue
        name, args = step["tool"], step.get("args", {})
        if name not in TOOLS:
            problems.append(f"Step {i}: unknown tool '{name}'. "
                            f"Tools: {', '.join(TOOLS)}")
            continue
        allowed = TOOLS[name][1]
        extra = set(args) - set(allowed)
        if extra:
            problems.append(f"Step {i}: '{name}' has no argument(s) {sorted(extra)}. "
                            f"Allowed: {list(allowed)}")
        for key in ("column", "by"):
            if key in args and args[key] not in columns:
                problems.append(f"Step {i}: unknown column '{args[key]}'. "
                                f"Columns: {', '.join(map(str, columns))}")
    return problems


def run_plan(question, plan, df):
    """Run validated steps in order. Stops at the first failing step."""
    report = Report(question, plan)
    problems = validate_plan(plan, list(df.columns))
    if problems:
        report.errors = problems
        return report
    current = df
    for i, step in enumerate(plan, 1):
        func = TOOLS[step["tool"]][0]
        result = func(current, **step.get("args", {}))
        if not result.ok:
            report.errors.append(f"Step {i} ({step['tool']}): {result.error}")
            return report
        if result.table is not None:
            report.tables.append((f"Step {i}: {result.note}", result.table))
        if result.frame is not None:
            current = result.frame
    return report


# ---------- optional: let a model propose the plan ----------

def _schema_text(df):
    return "\n".join(f"- {c} ({t})" for c, t in df.dtypes.astype(str).items())


def _tool_text():
    return "\n".join(f"- {name}({', '.join(args)})" for name, (_, args) in TOOLS.items())


def _parse_reply(text):
    """Return a list of steps, or a dict like {"cannot_answer": "reason"}."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    data = json.loads(m.group(0))
    if isinstance(data, dict) and "cannot_answer" in data:
        return data
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of steps")
    return data


RULES = ("Use only these tools and only these column names. Never substitute a "
         "different column for the one the question asks about. If the question "
         'cannot be answered with these columns, reply {"cannot_answer": "short reason"} '
         "instead of a plan.")


def plan_with_llm(question, df, llm, repairs=1):
    """Ask a model for a plan.

    Returns (plan, problems, cannot_answer). An invalid plan is sent back once
    with the errors. If the model says the question cannot be answered, that
    reason is returned and nothing is run.
    """
    prompt = (f"Question: {question}\n\nColumns:\n{_schema_text(df)}\n\n"
              f"Tools:\n{_tool_text()}\n\n"
              "Reply with only a JSON list of steps, each like "
              '{"tool": "group_summary", "args": {"by": "region", "column": "units", "agg": "sum"}}. '
              + RULES)
    reply = llm(prompt)
    plan, problems = [], []
    for attempt in range(repairs + 1):
        try:
            parsed = _parse_reply(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            plan, problems = [], [f"Could not read the reply as a JSON list: {exc}"]
        else:
            if isinstance(parsed, dict):
                return [], [], str(parsed["cannot_answer"])
            plan = parsed
            problems = validate_plan(plan, list(df.columns))
            if not problems:
                return plan, [], ""
        if attempt < repairs:
            reply = llm(prompt + "\n\nYour last plan had these problems:\n- "
                        + "\n- ".join(problems) + "\n" + RULES
                        + " Return a corrected JSON list, or cannot_answer.")
    return plan, problems, ""


def answer(question, df, llm):
    plan, problems, cannot = plan_with_llm(question, df, llm)
    report = Report(question, plan)
    if cannot:
        report.errors = ["This question cannot be answered with this table: " + cannot]
        return report
    if problems:
        report.errors = ["The model's plan could not be used: " + "; ".join(problems)]
        return report
    return run_plan(question, plan, df)
