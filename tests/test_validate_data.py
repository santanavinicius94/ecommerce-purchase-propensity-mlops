"""Testes do contrato de entrada com amostras pequenas e independentes."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.validate_data import COLUMNS, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.csv"
        self.row = ["0", "0", "0", "0", "1", "0", "0.2", "0.2", "0", "0",
                    "Feb", "1", "1", "1", "1", "Returning_Visitor", "FALSE", "FALSE"]

    def write_csv(self, rows, header=COLUMNS):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def test_counts_duplicates_and_preserves_bytes(self):
        purchase = self.row.copy()
        purchase[-1] = "TRUE"
        self.write_csv([self.row, purchase, self.row])
        original = self.path.read_bytes()
        report = validate_dataset(self.path)
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["target_counts"], {"FALSE": 2, "TRUE": 1})
        self.assertAlmostEqual(report["purchase_rate"], 1 / 3)
        self.assertEqual(original, self.path.read_bytes())

    def test_rejects_invalid_values(self):
        cases = [("Revenue", "yes"), ("Weekend", "0"), ("Month", "Foo"),
                 ("VisitorType", "Unknown"), ("PageValues", ""),
                 ("PageValues", "NaN"), ("PageValues", "inf"),
                 ("Administrative_Duration", "-1"), ("ProductRelated", "1.5"),
                 ("Browser", "0"), ("BounceRates", "1.1"), ("ExitRates", "abc")]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                row = self.row.copy()
                row[COLUMNS.index(column)] = value
                self.write_csv([row])
                with self.assertRaisesRegex(ValueError, column):
                    validate_dataset(self.path)

    def test_rejects_wrong_header_and_row_width(self):
        for header, row in [(COLUMNS[::-1], self.row), (COLUMNS, self.row[:-1]),
                            (COLUMNS, self.row + ["extra"])]:
            with self.subTest(header=header, row=row):
                self.write_csv([row], header)
                with self.assertRaises(ValueError):
                    validate_dataset(self.path)

    def test_rejects_empty_dataset(self):
        self.write_csv([])
        with self.assertRaisesRegex(ValueError, "sem registros"):
            validate_dataset(self.path)
        self.path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_dataset(self.path)

    def test_cli_returns_failure_for_missing_file(self):
        script = Path(__file__).resolve().parents[1] / "scripts/validate_data.py"
        result = subprocess.run([sys.executable, str(script), "--data", str(self.path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Erro", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
