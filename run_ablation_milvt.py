"""CLI runner for Table 6 and Table 7 ablation studies with real-time Drive sync."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from clat.ablation import (
    TABLE_6_CONFIGS,
    TABLE_7_CONFIGS,
    TABLE_6_METRICS,
    TABLE_7_METRICS,
    AblationExperimentRunner,
    AblationResultFormatter,
    DriveSyncManager,
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


def recompute_and_save_summary(
    configs: dict, metrics: list, start_fold: int, end_fold: int, csv_path: str, runner: AblationExperimentRunner
) -> list:
    """Recomputes statistics across all completed folds and saves CSV locally & to Drive."""
    summary_rows = []
    headers = ["Variant"] + [m.upper() for m in metrics]
    for v_name, _ in configs.items():
        exp_name = f"milvt_ablation_t6_{v_name}" if "6" in csv_path else f"milvt_ablation_t7_{v_name}"
        records = runner.collect_fold_metrics(exp_name, start_fold, end_fold)
        if records:
            stats = AblationResultFormatter.aggregate_statistics(records, metrics)
            summary_rows.append([v_name] + [stats[m] for m in metrics])
    if summary_rows:
        AblationResultFormatter.save_csv(csv_path, headers, summary_rows)
        runner.sync_manager.sync_summary_csv(csv_path)
    return summary_rows


def main() -> None:
    """Executes requested ablation variants with real-time fold sync and resume."""
    args = parse_arguments()
    configs = TABLE_6_CONFIGS if args.table == 6 else TABLE_7_CONFIGS
    metrics = TABLE_6_METRICS if args.table == 6 else TABLE_7_METRICS
    variants = list(configs.keys()) if args.variant == "all" else [args.variant]

    sync_mgr = DriveSyncManager()
    runner = AblationExperimentRunner(data_config=args.data, sync_manager=sync_mgr)
    csv_path = f"log/ablation_table_{args.table}_summary.csv"
    os.makedirs("log", exist_ok=True)

    for var in variants:
        cfg = configs[var]
        exp_name = f"milvt_ablation_t{args.table}_{var}"
        print(f"\n{'='*70}\n[STARTING] Table {args.table} Variant: {var} ({cfg})\n{'='*70}")
        for fold in range(args.start_fold, args.end_fold + 1):
            runner.execute_fold(cfg, exp_name, fold, dry_run=args.dry_run)
            recompute_and_save_summary(configs, metrics, args.start_fold, args.end_fold, csv_path, runner)

    headers = ["Variant"] + [m.upper() for m in metrics]
    summary_rows = recompute_and_save_summary(configs, metrics, args.start_fold, args.end_fold, csv_path, runner)
    rendered_md = AblationResultFormatter.render_markdown_table(headers, summary_rows)
    print(f"\n{'='*70}\n       FINAL ABLATION TABLE {args.table} RESULTS (Mean ± Std)\n{'='*70}")
    print(rendered_md)
    print(f"\n[Saved] Summary saved to {csv_path}\n{'='*70}\n")


if __name__ == "__main__":
    main()
