#!/usr/bin/env python3
"""
Score statistics script
Calculate total scores for a model across all levels
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_summary_report(report_path: Path) -> Optional[Dict[str, Any]]:
    """
    Read summary_report.json file
    Returns: Dictionary containing statistics
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        overall_stats = data.get('overall_statistics', {})
        
        return {
            'total_cases': overall_stats.get('total_cases', 0),
            'successful_cases': overall_stats.get('successful_cases', 0),
            'failed_cases': overall_stats.get('failed_cases', 0),
            'total_matched_products': overall_stats.get('total_matched_products', 0),
            'total_expected_products': overall_stats.get('total_expected_products', 0),
            'total_extra_products': overall_stats.get('total_extra_products', 0),
            'average_case_score': overall_stats.get('average_case_score', 0.0),
            'overall_match_rate': overall_stats.get('overall_match_rate', 0.0),
            'incomplete_cases': overall_stats.get('incomplete_cases', 0),
            'incomplete_rate': overall_stats.get('incomplete_rate', 0.0),
            'valid': overall_stats.get('valid', False),
        }
    except Exception as e:
        print(f"❌ Error reading {report_path}: {e}")
        return None


def _is_timestamp_folder(name: str) -> bool:
    return name.isdigit()


def _select_timestamp(model_dir: Path, timestamp: Optional[str]) -> Optional[str]:
    if timestamp is not None:
        return timestamp
    if not model_dir.is_dir():
        return None
    ts_candidates = [d.name for d in model_dir.iterdir() if d.is_dir() and _is_timestamp_folder(d.name)]
    if not ts_candidates:
        return None
    ts_candidates.sort(key=int, reverse=True)
    return ts_candidates[0]


def _load_level_data(model_name: str, ts_dir: Path, timestamp: str) -> Dict[int, Dict[str, Any]]:
    level_data: Dict[int, Dict[str, Any]] = {}
    for level in (1, 2, 3):
        report_path = ts_dir / f"level{level}" / "summary_report.json"
        if not report_path.exists():
            continue
        stats = read_summary_report(report_path)
        if stats is None:
            continue
        level_data[level] = {
            'folder_name': f"{model_name}/{timestamp}/level{level}",
            **stats
        }
    return level_data


def _aggregate_statistics(model_name: str, timestamp: str, level_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    total_cases_sum = sum(level_data[level]['total_cases'] for level in level_data.keys())
    successful_cases_sum = sum(level_data[level]['successful_cases'] for level in level_data.keys())
    failed_cases_sum = sum(level_data[level]['failed_cases'] for level in level_data.keys())
    total_matched_products_sum = sum(level_data[level]['total_matched_products'] for level in level_data.keys())
    total_expected_products_sum = sum(level_data[level]['total_expected_products'] for level in level_data.keys())
    total_extra_products_sum = sum(level_data[level]['total_extra_products'] for level in level_data.keys())
    incomplete_cases_sum = sum(level_data[level]['incomplete_cases'] for level in level_data.keys())

    weighted_avg_score = 0.0
    if total_cases_sum > 0:
        weighted_avg_score = sum(
            level_data[level]['average_case_score'] * level_data[level]['total_cases']
            for level in level_data.keys()
        ) / total_cases_sum

    successful_rate = successful_cases_sum / total_cases_sum if total_cases_sum > 0 else 0.0
    match_rate = total_matched_products_sum / total_expected_products_sum if total_expected_products_sum > 0 else 0.0
    incomplete_rate = incomplete_cases_sum / total_cases_sum if total_cases_sum > 0 else 0.0
    all_valid = all(level_data[level]['valid'] for level in level_data.keys())

    return {
        'model_name': model_name,
        'batch_timestamp': timestamp,
        'statistics_time': datetime.now().isoformat(),
        'levels': {
            f'level_{level}': {
                'folder_name': level_data[level]['folder_name'],
                'total_cases': level_data[level]['total_cases'],
                'successful_cases': level_data[level]['successful_cases'],
                'failed_cases': level_data[level]['failed_cases'],
                'total_matched_products': level_data[level]['total_matched_products'],
                'total_expected_products': level_data[level]['total_expected_products'],
                'total_extra_products': level_data[level]['total_extra_products'],
                'average_case_score': level_data[level]['average_case_score'],
                'overall_match_rate': level_data[level]['overall_match_rate'],
                'incomplete_cases': level_data[level]['incomplete_cases'],
                'incomplete_rate': level_data[level]['incomplete_rate'],
                'valid': level_data[level]['valid'],
            }
            for level in sorted(level_data.keys())
        },
        'total': {
            'total_cases': total_cases_sum,
            'successful_cases': successful_cases_sum,
            'failed_cases': failed_cases_sum,
            'total_matched_products': total_matched_products_sum,
            'total_expected_products': total_expected_products_sum,
            'total_extra_products': total_extra_products_sum,
            'successful_rate': successful_rate,
            'match_rate': match_rate,
            'weighted_average_case_score': weighted_avg_score,
            'incomplete_cases': incomplete_cases_sum,
            'incomplete_rate': incomplete_rate,
            'valid': all_valid,
            'levels_completed': sorted(level_data.keys()),
            'timestamp': timestamp,
        },
    }


def calculate_model_statistics(
    model_name: str,
    result_report_dir: Path,
    timestamp: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not result_report_dir.exists():
        print(f"❌ Error: Directory {result_report_dir} does not exist")
        return None

    model_dir = result_report_dir / model_name
    selected_timestamp = _select_timestamp(model_dir, timestamp)
    if selected_timestamp is None:
        print(f"❌ Error: No timestamp folder found under {model_dir}")
        return None

    ts_dir = model_dir / selected_timestamp
    level_data = _load_level_data(model_name, ts_dir, selected_timestamp)
    if not level_data:
        print(f"❌ Error: No summary_report.json found under {ts_dir}")
        return None

    missing_levels = {1, 2, 3} - set(level_data.keys())
    if missing_levels:
        print(f"⚠️  Warning: Missing levels under {ts_dir}: {sorted(missing_levels)}")

    return _aggregate_statistics(model_name, selected_timestamp, level_data)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate statistics for a model across all levels")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to calculate statistics for"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Batch timestamp under result_report/{model_name}/{timestamp}/ (optional)",
    )
    parser.add_argument(
        "--result_report_dir",
        type=str,
        default=None,
        help="Path to result_report directory (default: script_dir/result_report)"
    )
    
    args = parser.parse_args()
    
    # Determine root directory
    script_dir = Path(__file__).resolve().parent.parent
    
    # Set result_report directory
    if args.result_report_dir:
        result_report_dir = Path(args.result_report_dir)
        if result_report_dir.is_absolute():
            pass
        else:
            result_report_dir = script_dir / args.result_report_dir
    else:
        result_report_dir = script_dir / "result_report"
    
    print(f"\n{'='*80}")
    print(f"📊 Calculating Statistics for Model: {args.model_name}")
    print(f"{'='*80}")
    print(f"  Result report directory: {result_report_dir}")
    print()
    
    # Calculate statistics
    statistics = calculate_model_statistics(args.model_name, result_report_dir, timestamp=args.timestamp)
    
    if statistics is None:
        print(f"❌ Failed to calculate statistics for model {args.model_name}")
        sys.exit(1)
    
    # Save results to result_report directory
    batch_timestamp = statistics.get("batch_timestamp")
    output_file = result_report_dir / args.model_name / str(batch_timestamp) / "statistics.json"

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Statistics saved to: {output_file}")
        print()
        
        # Print summary
        print(f"{'='*80}")
        print(f"📊 Summary for {args.model_name}")
        print(f"{'='*80}")
        print(f"  Levels completed: {', '.join(map(str, statistics['total']['levels_completed']))}")
        print(f"  Total cases: {statistics['total']['total_cases']}")
        print(f"  Successful cases: {statistics['total']['successful_cases']}")
        print(f"  Failed cases: {statistics['total']['failed_cases']}")
        print(f"  Successful rate: {statistics['total']['successful_rate']:.4f} ({statistics['total']['successful_rate']:.2%})")
        print(f"  Match rate: {statistics['total']['match_rate']:.4f} ({statistics['total']['match_rate']:.2%})")
        print(f"  Weighted average case score: {statistics['total']['weighted_average_case_score']:.4f} ({statistics['total']['weighted_average_case_score']:.2%})")
        print(f"  Total matched products: {statistics['total']['total_matched_products']}/{statistics['total']['total_expected_products']}")
        print(f"  Total extra products: {statistics['total']['total_extra_products']}")
        print(f"  Incomplete cases: {statistics['total']['incomplete_cases']} ({statistics['total']['incomplete_rate']:.2%})")
        print(f"  Model valid: {statistics['total']['valid']} {'✅' if statistics['total']['valid'] else '❌'}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Failed to save statistics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

