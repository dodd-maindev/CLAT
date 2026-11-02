"""CLI runner for Table 6 and Table 7 ablation studies on MIL-VT backbone."""

import argparse
import os
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from clat.ablation import (
    TABLE_6_CONFIGS,
    TABLE_7_CONFIGS,
    TABLE_6_METRICS,
    TABLE_7_METRICS,
    AblationExperimentRunner,
    AblationResultFormatter,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for the ablation runner."""
    parser = argparse.ArgumentParser(description="OLAP MIL-VT Ablation Runner")
    parser.add_argument("--table", type=int, choices=[6, 7], required=True)
    parser.add_argument("--variant", type=str, default="all")
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=9)
    parser.add_argument("--data", type=str, default="configs/data/FGADDR.yaml")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Executes requested ablation study variants and displays aggregated tables."""
    args = parse_arguments()
    configs = TABLE_6_CONFIGS if args.table == 6 else TABLE_7_CONFIGS
    metrics = TABLE_6_METRICS if args.table == 6 else TABLE_7_METRICS
    variants = list(configs.keys()) if args.variant == "all" else [args.variant]

    runner = AblationExperimentRunner(data_config=args.data)
    summary_rows = []
    headers = ["Variant"] + [m.upper() for m in metrics]

    os.makedirs("log", exist_ok=True)
    for var in variants:
        cfg = configs[var]
        exp_name = f"milvt_ablation_t{args.table}_{var}"
        print(f"\n{'='*70}\n[STARTING] Table {args.table} Variant: {var} ({cfg})\n{'='*70}")
        for fold in range(args.start_fold, args.end_fold + 1):
            runner.execute_fold(cfg, exp_name, fold, dry_run=args.dry_run)

        records = runner.collect_fold_metrics(exp_name, args.start_fold, args.end_fold)
        stats = AblationResultFormatter.aggregate_statistics(records, metrics)
        summary_rows.append([var] + [stats[m] for m in metrics])

    rendered_md = AblationResultFormatter.render_markdown_table(headers, summary_rows)
    print(f"\n{'='*70}\n       FINAL ABLATION TABLE {args.table} RESULTS (Mean ± Std)\n{'='*70}")
    print(rendered_md)

    csv_path = f"log/ablation_table_{args.table}_summary.csv"
    AblationResultFormatter.save_csv(csv_path, headers, summary_rows)
    print(f"\n[Saved] Summary saved to {csv_path}\n{'='*70}\n")


if __name__ == "__main__":
    main()
