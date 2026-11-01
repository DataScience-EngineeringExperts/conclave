"""Offline checks for bounded, non-authorizing diagnostic output."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "diagnostic_verifier", Path(__file__).with_name("pr42-verify-gate3-streaming.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class DiagnosticTests(unittest.TestCase):
    def report(self):
        return {"Results": [{"Target": "test", "Class": "lang-pkgs", "Type": "python-pkg",
                             "Vulnerabilities": [{"Severity": "HIGH", "PkgName": "test-pkg",
                                                  "InstalledVersion": "1", "VulnerabilityID": "TEST-1",
                                                  "FixedVersion": "2", "Status": "affected"}]}]}

    def test_actual_count_and_tuple_not_assumed_348(self):
        report = self.report()
        result = verifier.finding_summary(report)
        self.assertEqual(result["raw_finding_count"], 1)
        self.assertEqual(result["raw_severity_counts"], {"HIGH": 1})
        row = result["high_critical_rows"][0]
        self.assertEqual(row["row"]["PkgPath"], "")
        self.assertEqual(len(row["row"]), 8)
        self.assertEqual((row["FixedVersion"], row["Status"]), ("2", "affected"))
        self.assertIs(result["dispositions_validated"], False)

    def test_duplicates_and_unknown_severity_preserved(self):
        report = self.report()
        rows = report["Results"][0]["Vulnerabilities"]
        rows.append(copy.deepcopy(rows[0]))
        rows.append({"Severity": "UNKNOWN"})
        result = verifier.finding_summary(report)
        self.assertEqual(result["raw_finding_count"], 3)
        self.assertEqual(result["raw_severity_counts"], {"HIGH": 2, "UNKNOWN": 1})
        self.assertEqual(len(result["high_critical_rows"]), 2)

    def test_no_vulnerabilities_has_zero_not_inferred_success(self):
        result = verifier.finding_summary({"Results": [{"Vulnerabilities": None}]})
        self.assertEqual(result["raw_finding_count"], 0)
        self.assertIs(result["dispositions_validated"], False)

    def test_too_many_rows_rejected(self):
        with self.assertRaises(ValueError):
            verifier.finding_summary({"Results": [{"Vulnerabilities": [{"Severity": "LOW"}]*5001}]})

    def test_malformed_empty_findings_rejected(self):
        with self.assertRaises(ValueError):
            verifier.finding_summary({"Results": [{"Vulnerabilities": {}}]})

    def test_wrong_tuple_type_or_oversized_value_rejected(self):
        for value in (False, "x"*4097):
            report = self.report()
            report["Results"][0]["Vulnerabilities"][0]["FixedVersion"] = value
            with self.assertRaises(ValueError): verifier.finding_summary(report)

    def test_serialized_output_limit(self):
        with self.assertRaises(ValueError): verifier.bounded_diagnostics({"data": "x"*(2*verifier.CHUNK)})
        self.assertEqual(verifier.bounded_diagnostics({"small": True}), {"small": True})

    def test_final_wrapper_included_in_byte_limit(self):
        diagnostics = {"data": "x"*(2*verifier.CHUNK-50)}
        verifier.bounded_diagnostics(diagnostics)
        with self.assertRaises(ValueError):
            verifier.serialize_result({"diagnostics": diagnostics, "wrapper": "x"*100})

    def test_exact_output_limit_includes_newline(self):
        overhead = len(verifier.serialize_result({"data": ""}).encode())
        fitting = {"data": "x"*(2*verifier.CHUNK-overhead)}
        self.assertEqual(len(verifier.serialize_result(fitting).encode()), 2*verifier.CHUNK)
        fitting["data"] += "x"
        with self.assertRaises(ValueError): verifier.serialize_result(fitting)


if __name__ == "__main__":
    unittest.main()
