"""Run the analyst on the sample sales table.

    py run_demo.py                 offline, hand-written plans
    py run_demo.py --llm           a model proposes the plans (asks for a key)
    py run_demo.py --csv my.csv    use your own file with the hand-written describe step
"""
import sys

from agent import answer, run_plan
from llm_client import ask_llm
from sample_data import make_sales
from tools import load_csv

QUESTIONS = [
    ("What is the total revenue per region?",
     [{"tool": "group_summary", "args": {"by": "region", "column": "revenue", "agg": "sum"}}]),
    ("Which three laptop orders had the most units?",
     [{"tool": "filter_rows", "args": {"column": "product", "op": "==", "value": "laptop"}},
      {"tool": "top_n", "args": {"column": "units", "n": 3}}]),
    ("What is the average price per colour?",   # no such column: shows the error path
     [{"tool": "group_summary", "args": {"by": "colour", "column": "price", "agg": "mean"}}]),
]


def show(report):
    print(f"\nQ: {report.question}")
    for note, table in report.tables:
        print(f"  {note}")
        print(table.to_string(index=False))
    for err in report.errors:
        print(f"  ERROR: {err}")


def main():
    if "--csv" in sys.argv:
        df, err = load_csv(sys.argv[sys.argv.index("--csv") + 1])
        if err:
            sys.exit(err)
        show(run_plan("Describe the file", [{"tool": "describe"}], df))
        return
    df = make_sales()
    use_llm = "--llm" in sys.argv
    for question, plan in QUESTIONS:
        show(answer(question, df, ask_llm) if use_llm else run_plan(question, plan, df))


if __name__ == "__main__":
    main()
