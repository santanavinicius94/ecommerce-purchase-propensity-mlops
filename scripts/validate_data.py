"""Valida o contrato do CSV e apresenta um perfil sem alterar os dados."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/raw/online_shoppers_intention.csv"
COLUMNS = (
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend", "Revenue",
)
COUNTS = {"Administrative", "Informational", "ProductRelated"}
CODES = {"OperatingSystems", "Browser", "Region", "TrafficType"}
RATES = {"BounceRates", "ExitRates", "SpecialDay"}
CATEGORIES = {
    "Month": {"Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"},
    "VisitorType": {"New_Visitor", "Returning_Visitor", "Other"},
    "Weekend": {"TRUE", "FALSE"},
    "Revenue": {"TRUE", "FALSE"},
}


def validate_value(column: str, value: str) -> None:
    """Rejeita ausências e valores fora do domínio inicial."""
    if not value.strip():
        raise ValueError(f"{column}: valor vazio")
    if column in CATEGORIES:
        if value not in CATEGORIES[column]:
            raise ValueError(f"{column}: categoria inválida {value!r}")
        return
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{column}: número inválido {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{column}: esperado número finito não negativo")
    if column in COUNTS | CODES and not number.is_integer():
        raise ValueError(f"{column}: esperado inteiro")
    if column in CODES and number < 1:
        raise ValueError(f"{column}: esperado código positivo")
    if column in RATES and number > 1:
        raise ValueError(f"{column}: esperado valor entre 0 e 1")


def validate_dataset(path: Path) -> dict:
    """Valida todas as linhas; duplicatas são informadas, nunca removidas."""
    target = Counter()
    seen = set()
    duplicates = 0
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, [])) != COLUMNS:
            raise ValueError("Cabeçalho diferente do esquema esperado (nomes e ordem)")
        for row in reader:
            if len(row) != len(COLUMNS):
                raise ValueError(f"Linha {reader.line_num}: quantidade incorreta de colunas")
            for column, value in zip(COLUMNS, row):
                try:
                    validate_value(column, value)
                except ValueError as error:
                    raise ValueError(f"Linha {reader.line_num}: {error}") from error
            key = tuple(row)
            duplicates += key in seen
            seen.add(key)
            target[row[-1]] += 1
            rows += 1
    if not rows:
        raise ValueError("Arquivo sem registros")
    return {
        "rows": rows,
        "columns": len(COLUMNS),
        "missing_values": 0,
        "duplicate_rows": duplicates,
        "target_counts": {label: target[label] for label in ("FALSE", "TRUE")},
        "purchase_rate": target["TRUE"] / rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Caminho do CSV")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.data)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Erro de validação: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
