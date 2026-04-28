#!/usr/bin/env python3
"""
EquiLens AI — Fairness Analysis Demo Script

This script demonstrates a complete end-to-end fairness audit:

1. Load sample loan dataset
2. Run full analysis pipeline
3. Print clean JSON output
4. Show privacy guarantees

Usage:
    # Default: Use sample data
    python analyze_demo.py

    # Custom CSV
    python analyze_demo.py --dataset /path/to/data.csv

    # Without Gemini explanation (local-only analysis)
    python analyze_demo.py --no-explanation

Environment:
    GEMINI_API_KEY=xxx python analyze_demo.py  # For AI explanations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from backend.services.full_analysis_pipeline import run_full_analysis_sync

# Force UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =====================================================================
# Logging Setup
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for demo script."""
    parser = argparse.ArgumentParser(
        description="EquiLens AI — Fairness Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Load sample data and run analysis with Gemini
  python analyze_demo.py

  # Use custom dataset
  python analyze_demo.py --dataset data/my_loans.csv

  # Local-only (no Gemini)
  python analyze_demo.py --no-explanation

  # With debug logging
  python analyze_demo.py --debug
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV dataset (default: frontend/sample_data.csv)",
    )

    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="Skip Gemini explanation (local analysis only)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON output to file (optional)",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Resolve dataset path
    # ------------------------------------------------------------------
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {args.dataset}")
            sys.exit(1)
    else:
        # Default to sample data in frontend/
        dataset_path = Path(__file__).parent / "frontend" / "sample_data.csv"
        if not dataset_path.exists():
            logger.error(f"Default sample dataset not found: {dataset_path}")
            logger.info("Provide --dataset to use a custom CSV file")
            sys.exit(1)

    logger.info(f"Using dataset: {dataset_path}")

    # ------------------------------------------------------------------
    # Run full analysis
    # ------------------------------------------------------------------
    try:
        result = run_full_analysis_sync(
            file_path=dataset_path,
            protected_cols=["gender"],  # Use gender (binary: M/F)
            target_col="approved",
            prediction_col="predicted",
            model_name="loan_approval_model",
            use_case="loan_approval",
            application_domain="lending",
            regulatory_requirements="Fair Lending, EU AI Act",
            generate_explanation=not args.no_explanation,
            min_group_size=5,  # Demo: Allow smaller groups for sample data
        )

        # ------------------------------------------------------------------
        # Output results
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("✓ FAIRNESS AUDIT COMPLETE")
        print("=" * 80)

        # Print JSON output
        json_output = json.dumps(result, indent=2)
        print(json_output)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                f.write(json_output)
            logger.info(f"✓ Results saved to: {output_path}")

        # ------------------------------------------------------------------
        # Privacy guarantees summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("🔐 PRIVACY GUARANTEES")
        print("=" * 80)
        print("✓ No raw dataset rows sent to external APIs")
        print("✓ No personal identifiers in explanation payload")
        print("✓ Only aggregated metrics (group statistics) sent to Gemini")
        print("✓ All fairness metrics computed locally (deterministic)")
        print("✓ Privacy validation performed before external API calls")

        # ------------------------------------------------------------------
        # Summary statistics
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"Report ID:         {result.get('report_id')}")
        print(f"Model:             {result.get('model_name')}")
        print(f"Dataset rows:      {result.get('dataset_shape', {}).get('rows')}")
        print(f"Overall Fair:      {result.get('overall', {}).get('is_fair')}")
        print(f"Protected attrs:   {', '.join(result.get('protected_attributes', []))}")

        # Per-attribute fairness
        per_attr = result.get("per_attribute", {})
        if per_attr:
            print("\nPer-Attribute Fairness:")
            for attr, attr_data in per_attr.items():
                is_fair = attr_data.get("is_fair", False)
                status = "✓ FAIR" if is_fair else "✗ BIASED"
                print(f"  {attr:20s} {status}")

        print("\n" + "=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
