import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


TOTAL_IDS = 120


def _split_words(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return value.split()


def _count_missing_reports(reports_dir: Path) -> int:
    if not reports_dir.is_dir():
        return TOTAL_IDS
    existing_ids: set[int] = set()
    for f in reports_dir.glob("id_*.txt"):
        try:
            id_num = int(f.stem.split("_")[1])
            existing_ids.add(id_num)
        except Exception:
            pass
    return len(sorted(set(range(TOTAL_IDS)) - existing_ids))


def _count_missing_plans(plans_dir: Path) -> int:
    if not plans_dir.is_dir():
        return TOTAL_IDS
    existing_ids: set[int] = set()
    for f in plans_dir.glob("id_*_converted.json"):
        try:
            id_num = int(f.stem.split("_")[1])
            existing_ids.add(id_num)
        except Exception:
            pass
    return len(sorted(set(range(TOTAL_IDS)) - existing_ids))


def _tail_completion_line(log_file: Path) -> str | None:
    if not log_file.is_file():
        return None
    pattern = re.compile(r"Model.*Language.*completed")
    last: str | None = None
    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if pattern.search(line):
                last = line
    return last


def _chmod_best_effort(target_dir: Path) -> None:
    try:
        for p in target_dir.rglob("*"):
            try:
                if p.is_dir():
                    os.chmod(p, 0o700)
                else:
                    os.chmod(p, 0o600)
            except Exception:
                pass
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Travel Planning Concurrent Runner (Windows-friendly)",allow_abbrev=False)
    parser.add_argument(
        "--models",
        type=str,
        default="minimax-m2.5",
        help='Space-separated model config names, e.g. "qwen-plus gpt-4o-2024-11-20"',
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="",
        help='IDs to run (passed to run.py --rerun-ids). Examples: "5", "3,17,42", "0-10,15". Empty means all.',
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help='Language: "zh", "en", or empty string for both (pass --language "")',
    )
    parser.add_argument("--workers", type=int, default=40, help="Parallel workers")
    parser.add_argument("--max-llm-calls", type=int, default=200, help="Max LLM calls per task")
    parser.add_argument(
        "--start-from",
        type=str,
        default="inference",
        choices=["inference", "conversion", "evaluation"],
        help="Start point: inference, conversion, evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (empty means default results/...)",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="",
        help='Verbose mode value (non-empty adds "--verbose" to run.py)',
    )
    parser.add_argument(
        "--debug",
        type=str,
        default="",
        help='Debug mode value (non-empty adds "--debug" to run.py)',
    )
    args = parser.parse_args()
    if args.language not in ("", "zh", "en"):
        raise ValueError('language must be "zh", "en", or ""')
    return args


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    args = _parse_args()
    task_id_value = args.task_id
    language_value = args.language
    workers_value = args.workers
    max_llm_calls_value = args.max_llm_calls
    start_from_value = args.start_from
    output_dir_value = args.output_dir
    verbose_value = args.verbose
    debug_value = args.debug

    models = _split_words(args.models)

    if language_value == "":
        language_display = "zh + en (both languages)"
    else:
        language_display = language_value

    log_dir = Path(tempfile.mkdtemp())

    processes: list[subprocess.Popen] = []
    pid_to_model: dict[int, str] = {}

    shutting_down = False

    def handle_sigint(_sig, _frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print("🛑 Caught Ctrl+C, stopping all tasks...")
        for p in processes:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in processes:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        try:
            shutil.rmtree(log_dir, ignore_errors=True)
        except Exception:
            pass
        raise SystemExit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    print("================================")
    print("🚀 Starting Concurrent Evaluation")
    print(f"Total Models in config: {len(models)}")
    print(f"Language: {language_display}")
    print(f"Workers: {workers_value}")
    print(f"Start From: {start_from_value}")
    print("================================")
    print("")

    models_to_run: list[str] = []
    model_skip_reason: dict[str, str] = {}
    model_start_from: dict[str, str] = {}

    if start_from_value == "inference":
        print("🔍 Pre-check: Detecting missing reports and converted plans for each model...")
        print("")

        for model_name in models:
            should_skip = False
            skip_reason = ""
            model_start = "inference"

            if language_value == "":
                zh_reports_missing = 0
                en_reports_missing = 0
                zh_plans_missing = 0
                en_plans_missing = 0

                for lang in ("zh", "en"):
                    if output_dir_value != "":
                        reports_dir = Path(output_dir_value) / f"{model_name}_{lang}" / "reports"
                        plans_dir = Path(output_dir_value) / f"{model_name}_{lang}" / "converted_plans"
                    else:
                        reports_dir = Path("results") / f"{model_name}_{lang}" / "reports"
                        plans_dir = Path("results") / f"{model_name}_{lang}" / "converted_plans"

                    reports_missing = _count_missing_reports(reports_dir)
                    plans_missing = _count_missing_plans(plans_dir)

                    if reports_missing == 0 and plans_missing == 0:
                        print(f"  ✅ {model_name} ({lang}): All complete (reports + plans)")
                    elif reports_missing == 0 and plans_missing > 0:
                        print(f"  📝 {model_name} ({lang}): Reports ✅ | Plans: {plans_missing} missing")
                    elif reports_missing > 0:
                        print(f"  📝 {model_name} ({lang}): Reports: {reports_missing} missing | Plans: {plans_missing} missing")

                    if lang == "zh":
                        zh_reports_missing = reports_missing
                        zh_plans_missing = plans_missing
                    else:
                        en_reports_missing = reports_missing
                        en_plans_missing = plans_missing

                if (
                    zh_reports_missing == 0
                    and en_reports_missing == 0
                    and zh_plans_missing == 0
                    and en_plans_missing == 0
                ):
                    should_skip = True
                    skip_reason = "Both zh and en have all reports and plans"
                elif zh_reports_missing == 0 and en_reports_missing == 0:
                    model_start = "conversion"
                else:
                    model_start = "inference"
            else:
                if output_dir_value != "":
                    reports_dir = Path(output_dir_value) / f"{model_name}_{language_value}" / "reports"
                    plans_dir = Path(output_dir_value) / f"{model_name}_{language_value}" / "converted_plans"
                else:
                    reports_dir = Path("results") / f"{model_name}_{language_value}" / "reports"
                    plans_dir = Path("results") / f"{model_name}_{language_value}" / "converted_plans"

                reports_missing = _count_missing_reports(reports_dir)
                plans_missing = _count_missing_plans(plans_dir)

                if reports_missing == 0 and plans_missing == 0:
                    print(f"  ✅ {model_name}: All complete (reports + plans)")
                    should_skip = True
                    skip_reason = f"All reports and plans exist for language {language_value}"
                elif reports_missing == 0 and plans_missing > 0:
                    print(f"  📝 {model_name}: Reports ✅ | Plans: {plans_missing} missing")
                    model_start = "conversion"
                elif reports_missing > 0:
                    print(f"  📝 {model_name}: Reports: {reports_missing} missing | Plans: {plans_missing} missing")
                    model_start = "inference"

            if should_skip:
                model_skip_reason[model_name] = skip_reason
            else:
                models_to_run.append(model_name)
                model_start_from[model_name] = model_start

        print("")

        if len(model_skip_reason) > 0:
            print("⏭️  Skipping models (already complete):")
            for model_name, reason in model_skip_reason.items():
                print(f"   - {model_name}: {reason}")
            print("")
    else:
        models_to_run = list(models)
        for model_name in models:
            model_start_from[model_name] = start_from_value

    total = len(models_to_run)

    if total == 0:
        print("✅ All models are already complete. Nothing to run!")
        return 0

    inference_count = 0
    conversion_count = 0
    for model_name in models_to_run:
        if model_start_from.get(model_name) == "conversion":
            conversion_count += 1
        else:
            inference_count += 1

    print("================================")
    print(f"📊 Will run {total} models (skipped {len(model_skip_reason)})")
    if inference_count > 0:
        print(f"   - From inference: {inference_count} models")
    if conversion_count > 0:
        print(f"   - From conversion: {conversion_count} models (reports complete, only convert plans)")
    print("================================")
    print("")

    if output_dir_value != "" and Path(output_dir_value).is_dir():
        print("🔧 Fixing permissions for output directory...")
        _chmod_best_effort(Path(output_dir_value))
        print("   ✅ Permissions fixed")
        print("")

    for model_name in models_to_run:
        log_file = log_dir / f"{model_name}.log"
        model_start = model_start_from.get(model_name, start_from_value)

        print(f"[STARTED] {model_name} (start-from: {model_start}) ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"   📝 Log: {log_file}")

        cmd: list[str] = [
            sys.executable,
            "-X",
            "utf8",
            "run.py",
            "--model",
            model_name,
            "--workers",
            str(workers_value),
            "--max-llm-calls",
            str(max_llm_calls_value),
            "--start-from",
            model_start,
        ]

        if language_value != "":
            cmd += ["--language", language_value]
        if output_dir_value != "":
            cmd += ["--output-dir", output_dir_value]

        if task_id_value != "":
            cmd += ["--rerun-ids", task_id_value]

        if verbose_value != "":
            cmd += ["--verbose"]
        if debug_value != "":
            cmd += ["--debug"]

        log_fp = log_file.open("w", encoding="utf-8", errors="replace")
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        p = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, cwd=script_dir, env=env)
        p._eval_log_fp = log_fp  # type: ignore[attr-defined]

        processes.append(p)
        pid_to_model[p.pid] = model_name

    print("")
    print("All models started, waiting for completion...")
    print("")

    completed = 0
    success = 0
    failed = 0
    failed_models: list[str] = []
    processed_pids: set[int] = set()

    while completed < total:
        for p in processes:
            if p.pid in processed_pids:
                continue
            rc = p.poll()
            if rc is None:
                continue

            processed_pids.add(p.pid)

            model_name = pid_to_model.get(p.pid)
            if model_name is None:
                continue

            try:
                log_fp = getattr(p, "_eval_log_fp", None)
                if log_fp is not None:
                    log_fp.close()
            except Exception:
                pass

            exit_path = log_dir / f"{model_name}.exit"
            try:
                exit_path.write_text(str(rc), encoding="utf-8")
            except Exception:
                pass

            exit_code = rc
            if exit_path.is_file():
                try:
                    file_exit = exit_path.read_text(encoding="utf-8").strip()
                    if file_exit != "":
                        exit_code = int(file_exit)
                except Exception:
                    pass

            completed += 1

            if exit_code == 0:
                success += 1
                print(f"[{completed}/{total}] ✅ {model_name} - Completed Successfully ({time.strftime('%H:%M:%S')})")
                completion_line = _tail_completion_line(log_dir / f"{model_name}.log")
                if completion_line:
                    print(f"   {completion_line}")
            else:
                failed += 1
                failed_models.append(model_name)
                print(f"[{completed}/{total}] ❌ {model_name} - Failed (exit code: {exit_code}) ({time.strftime('%H:%M:%S')})")
                print(f"   See log: {log_dir / f'{model_name}.log'}")

        if completed < total:
            time.sleep(1)

    print("")
    print("================================")
    print("📊 BATCH EVALUATION SUMMARY")
    print(f"Total: {total} | Success: {success} | Failed: {failed}")
    if failed > 0:
        print(f"Failed models: {' '.join(failed_models)}")
        print("")
        print(f"Log directory: {log_dir}")
    else:
        shutil.rmtree(log_dir, ignore_errors=True)
    print("================================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

