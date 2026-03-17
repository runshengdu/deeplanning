import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _split_words(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return value.split()


def _generate_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.getpid()}_{random.randint(0, 32767)}"


def _copy_contents(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        src = item
        dst = dst_dir / item.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _parse_id_list(id_str: str | None) -> list[int] | None:
    if not id_str:
        return None

    ids: set[int] = set()
    parts = id_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start <= end:
                    ids.update(range(start, end + 1))
                else:
                    ids.update(range(end, start + 1))
            except ValueError:
                print(f"⚠️  Warning: Invalid range format '{part}', skipping")
        else:
            try:
                ids.add(int(part))
            except ValueError:
                print(f"⚠️  Warning: Invalid ID '{part}', skipping")

    return sorted(ids)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shopping Benchmark Runner (Windows-friendly)",allow_abbrev=False)
    parser.add_argument(
        "--levels",
        default="1 2 3",
        help='Space-separated levels to run, e.g. "1" or "1 2 3"',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=100,
        help="Maximum LLM calls per sample",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="minimax-m2.5",
        help='Space-separated model config names, e.g. "qwen-plus gpt-4o-2024-11-20"',
    )
    parser.add_argument(
        "--rerun-ids",
        type=str,
        default="",
        help='Comma-separated list of IDs to run (e.g., "3,17,42" or "0-10,15"). Empty means run all.',
    )
    return parser.parse_args()


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    args = _parse_args()
    models = _split_words(args.models)
    levels = _split_words(args.levels)
    rerun_ids = _parse_id_list(args.rerun_ids)

    for model in models:
        os.environ["SHOPPING_AGENT_MODEL"] = model
        batch_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for test_level in levels:
            os.chdir(base_dir)

            run_id = _generate_run_id()
            database_run_dir = f"database_run_{model}_level{test_level}_{run_id}"

            print("")
            print("🚀 Starting Shopping Benchmark")
            print(f"   Level: {test_level}")
            print(f"   Model: {os.environ.get('SHOPPING_AGENT_MODEL')}")
            print(f"   Workers: {args.workers}")
            print(f"   Max LLM calls: {args.max_llm_calls}")
            if rerun_ids is not None:
                print(f"   Rerun IDs: {args.rerun_ids}")
            print(f"   Database: {database_run_dir}")
            print("")

            shutil.rmtree(database_run_dir, ignore_errors=True)
            Path(database_run_dir).mkdir(parents=True, exist_ok=True)

            src_db = base_dir / "database" / f"database_level{test_level}"
            if not src_db.is_dir():
                raise FileNotFoundError(str(src_db))
            _copy_contents(src_db, Path(database_run_dir))
            print(f"📁 Created isolated database: {database_run_dir}")

            cmd = [
                sys.executable,
                "run.py",
                "--model",
                model,
                "--workers",
                str(args.workers),
                "--level",
                str(test_level),
                "--max-llm-calls",
                str(args.max_llm_calls),
                "--database-dir",
                database_run_dir,
            ]
            if rerun_ids is not None:
                cmd.extend(["--rerun-ids", args.rerun_ids])

            result = subprocess.run(cmd, cwd=base_dir)
            exit_code = result.returncode
            if exit_code != 0:
                print(f"❌ Inference failed for model {os.environ.get('SHOPPING_AGENT_MODEL')} level {test_level}")
                shutil.rmtree(database_run_dir, ignore_errors=True)
                return 1

            print("<<<< Starting Evaluation >>>>")
            output_folder = f"{os.environ.get('SHOPPING_AGENT_MODEL')}/{batch_timestamp}/level{test_level}"
            database_dst = base_dir / "database_infered" / output_folder
            database_dst.parent.mkdir(parents=True, exist_ok=True)
            if database_dst.exists():
                shutil.rmtree(database_dst, ignore_errors=True)
            shutil.move(database_run_dir, str(database_dst))
            print(f"<<<< Database saved to: database_infered/{output_folder} >>>>")

            eval_cmd = [sys.executable, "evaluation/evaluation_pipeline.py", "--database_dir", output_folder]
            if rerun_ids is not None:
                case_filter = [f"case_{i}" for i in rerun_ids]
                if case_filter:
                    eval_cmd.extend(["--case_filter", *case_filter])

            subprocess.run(
                eval_cmd,
                cwd=base_dir,
                check=True,
            )

            print(f"✅ Model {os.environ.get('SHOPPING_AGENT_MODEL')} Level {test_level} finished.")

            if test_level != levels[-1]:
                print("Sleeping 10s before next level...")
                time.sleep(10)

        print("")
        print(f"<<<< Calculating Overall Statistics for {os.environ.get('SHOPPING_AGENT_MODEL')} >>>>")
        stats_result = subprocess.run(
            [
                sys.executable,
                "evaluation/score_statistics.py",
                "--model_name",
                os.environ.get("SHOPPING_AGENT_MODEL", ""),
                "--timestamp",
                batch_timestamp,
            ],
            cwd=base_dir,
        )
        if stats_result.returncode != 0:
            print(f"⚠️  Warning: Statistics calculation failed for model {os.environ.get('SHOPPING_AGENT_MODEL')}, continuing...")
        else:
            print(f"✅ Statistics calculation completed for {os.environ.get('SHOPPING_AGENT_MODEL')}")
        print("")

        if model != models[-1]:
            print("Sleeping 60s before next model...")
            time.sleep(60)

    print("")
    print("✅ All models completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
