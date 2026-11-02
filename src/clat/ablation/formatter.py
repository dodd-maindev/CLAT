"""Formats and aggregates ablation experiment results into paper-ready tables."""

import csv
import math
import statistics
from typing import Any, Dict, List


class AblationResultFormatter:
    """Computes mean and std statistics and formats paper-ready comparison tables."""

    @staticmethod
    def aggregate_statistics(
        records: List[Dict[str, Any]], metrics: List[str]
    ) -> Dict[str, str]:
        """Calculates mean ± std formatted string for each metric."""
        results: Dict[str, str] = {}
        for m in metrics:
            vals = [
                float(r[m]) * 100.0
                for r in records
                if m in r and r[m] is not None and not math.isnan(float(r[m]))
            ]
            if not vals:
                results[m] = "—"
            elif len(vals) == 1:
                results[m] = f"{vals[0]:.2f}"
            else:
                mean_val = statistics.mean(vals)
                std_val = statistics.stdev(vals)
                results[m] = f"{mean_val:.2f} ± {std_val:.2f}"
        return results

    @staticmethod
    def render_markdown_table(
        headers: List[str], rows: List[List[str]]
    ) -> str:
        """Renders an ASCII markdown table from headers and row data."""
        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(str(cell)))

        header_line = "| " + " | ".join(
            f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))
        ) + " |"
        sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
        lines = [header_line, sep_line]
        for row in rows:
            lines.append(
                "| " + " | ".join(
                    f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))
                ) + " |"
            )
        return "\n".join(lines)

    @staticmethod
    def save_csv(path: str, headers: List[str], rows: List[List[str]]) -> None:
        """Writes table data to CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
