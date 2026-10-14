import os
import tempfile
import unittest

import pandas as pd

from agent import answer, plan_with_llm, run_plan, validate_plan
from sample_data import make_sales
from tools import describe, filter_rows, group_summary, load_csv, top_n, value_counts


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.df = make_sales()

    def test_group_summary_sums_correctly(self):
        res = group_summary(self.df, "region", "revenue", "sum")
        self.assertTrue(res.ok)
        self.assertEqual(int(res.table.iloc[:, 1].sum()), int(self.df["revenue"].sum()))

    def test_unknown_column_gives_helpful_error(self):
        res = group_summary(self.df, "colour", "price", "mean")
        self.assertFalse(res.ok)
        self.assertIn("Available columns", res.error)

    def test_mean_of_text_column_is_rejected(self):
        res = group_summary(self.df, "region", "product", "mean")
        self.assertIn("not numeric", res.error)

    def test_filter_keeps_matching_rows_and_passes_frame_on(self):
        res = filter_rows(self.df, "product", "==", "laptop")
        self.assertTrue((res.frame["product"] == "laptop").all())
        self.assertLess(len(res.frame), len(self.df))

    def test_filter_numeric_value_given_as_text(self):
        res = filter_rows(self.df, "units", ">", "5")
        self.assertTrue((res.frame["units"] > 5).all())

    def test_filter_rejects_non_numeric_value_for_numeric_column(self):
        self.assertIn("not a number", filter_rows(self.df, "units", ">", "many").error)

    def test_text_column_rejects_greater_than(self):
        self.assertIn("numeric", filter_rows(self.df, "region", ">", "A").error)

    def test_top_n_orders_descending(self):
        res = top_n(self.df, "revenue", 3)
        self.assertEqual(list(res.table["revenue"]),
                         sorted(self.df["revenue"], reverse=True)[:3])

    def test_top_n_bad_n(self):
        self.assertFalse(top_n(self.df, "revenue", 0).ok)
        self.assertFalse(top_n(self.df, "revenue", "x").ok)

    def test_describe_and_value_counts(self):
        self.assertIn("missing", describe(self.df).table.columns)
        self.assertEqual(int(value_counts(self.df, "region").table["count"].sum()), 60)

    def test_missing_values_are_reported(self):
        df = self.df.copy()
        df.loc[0, "units"] = None
        t = describe(df).table
        self.assertEqual(int(t.loc[t["column"] == "units", "missing"].iloc[0]), 1)


class LoadTests(unittest.TestCase):
    def test_missing_file(self):
        df, err = load_csv("does_not_exist.csv")
        self.assertIsNone(df)
        self.assertIn("not found", err.lower())

    def test_empty_and_header_only_files(self):
        with tempfile.TemporaryDirectory() as d:
            empty, header = os.path.join(d, "e.csv"), os.path.join(d, "h.csv")
            open(empty, "w").close()
            with open(header, "w") as fh:
                fh.write("a,b\n")
            self.assertIn("empty", load_csv(empty)[1])
            self.assertIn("no rows", load_csv(header)[1])


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.df = make_sales()

    def test_validation_catches_unknown_tool_arg_and_column(self):
        plan = [{"tool": "magic"},
                {"tool": "top_n", "args": {"column": "revenue", "limit": 3}},
                {"tool": "top_n", "args": {"column": "profit"}}]
        problems = validate_plan(plan, list(self.df.columns))
        self.assertEqual(len(problems), 3)

    def test_filter_then_top_n_runs_on_the_filtered_rows(self):
        plan = [{"tool": "filter_rows", "args": {"column": "product", "op": "==", "value": "laptop"}},
                {"tool": "top_n", "args": {"column": "units", "n": 3}}]
        report = run_plan("q", plan, self.df)
        self.assertTrue(report.ok)
        self.assertTrue((report.tables[-1][1]["product"] == "laptop").all())

    def test_run_stops_at_first_failing_step(self):
        plan = [{"tool": "filter_rows", "args": {"column": "units", "op": ">", "value": "many"}},
                {"tool": "describe"}]
        report = run_plan("q", plan, self.df)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.tables), 0)
        self.assertIn("Step 1", report.errors[0])

    def test_invalid_plan_is_not_executed(self):
        report = run_plan("q", [{"tool": "describe", "args": {"bogus": 1}}], self.df)
        self.assertFalse(report.ok)
        self.assertEqual(report.tables, [])


class LlmPlannerTests(unittest.TestCase):
    def setUp(self):
        self.df = make_sales()

    def test_prompt_contains_columns_but_no_rows(self):
        seen = []
        good = '[{"tool": "describe"}]'
        plan_with_llm("q", self.df, lambda p: seen.append(p) or good)
        self.assertIn("revenue", seen[0])
        self.assertNotIn("laptop", seen[0])          # a value that only appears in rows
        self.assertNotIn("2026-01", seen[0])

    def test_bad_plan_is_repaired_once(self):
        replies = iter(['[{"tool": "group_summary", "args": {"by": "colour", "column": "price"}}]',
                        '[{"tool": "group_summary", "args": {"by": "region", "column": "price", "agg": "mean"}}]'])
        calls = []
        def fake(prompt):
            calls.append(prompt)
            return next(replies)
        plan, problems, cannot = plan_with_llm("avg price per region", self.df, fake)
        self.assertEqual(problems, [])
        self.assertEqual(len(calls), 2)
        self.assertIn("unknown column 'colour'", calls[1])
        self.assertIn("cannot_answer", calls[1])

    def test_unusable_reply_gives_clear_error_not_a_crash(self):
        report = answer("q", self.df, lambda p: "I cannot help with that")
        self.assertFalse(report.ok)
        self.assertIn("could not be used", report.errors[0])

    def test_model_can_decline_instead_of_substituting_a_column(self):
        report = answer("average price per colour", self.df,
                        lambda p: '{"cannot_answer": "there is no colour column"}')
        self.assertFalse(report.ok)
        self.assertEqual(report.tables, [])
        self.assertIn("cannot be answered", report.errors[0])
        self.assertIn("no colour column", report.errors[0])

    def test_decline_after_a_bad_first_plan(self):
        replies = iter(['[{"tool": "group_summary", "args": {"by": "colour", "column": "price"}}]',
                        '{"cannot_answer": "no colour column"}'])
        report = answer("q", self.df, lambda p: next(replies))
        self.assertIn("cannot be answered", report.errors[0])

    def test_the_prompt_tells_the_model_not_to_substitute_columns(self):
        seen = []
        plan_with_llm("q", self.df, lambda p: seen.append(p) or '[{"tool": "describe"}]')
        self.assertIn("Never substitute", seen[0])
        self.assertIn("cannot_answer", seen[0])


if __name__ == "__main__":
    unittest.main()
