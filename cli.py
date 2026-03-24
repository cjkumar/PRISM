"""
PRISM Command-Line Interface
=============================

Entry point for running PRISM analysis from the command line.

Usage:
    # Analyze a single document (cancer plan)
    python -m PRISM.cli analyze --pdf plan.pdf --country "Australia" --year 2023

    # Analyze a single document (CVD plan)
    python -m PRISM.cli analyze --pdf plan.pdf --country "Brazil" --year 2021 --domain cvd

    # Batch analyze all PDFs in a folder
    python -m PRISM.cli batch --input-dir ./pdfs --domain cancer

    # Validate existing analysis JSON files
    python -m PRISM.cli validate --folder ./NCCP_Analyses

    # Export JSON analyses to CSV
    python -m PRISM.cli export --folder ./NCCP_Analyses --output ./nccp_data

    # Generate summary statistics
    python -m PRISM.cli summary --folder ./NCCP_Analyses --domain cancer
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("prism.cli")


def cmd_analyze(args):
    """Run the full PRISM pipeline on a single document."""
    from PRISM.config import PRISMConfig
    from PRISM.pipeline import PRISMPipeline

    config_kwargs = {"domain": args.domain}
    if args.config:
        config = PRISMConfig.load(args.config)
    else:
        config = PRISMConfig(**config_kwargs)

    if args.output_dir:
        config.output_dir = args.output_dir

    pipeline = PRISMPipeline(config)
    result = pipeline.process_document(
        pdf_path=args.pdf,
        country=args.country,
        year=args.year,
        output_path=args.output,
        lightweight_ingestion=args.lightweight,
    )

    print(f"\nAnalysis complete: {result['country']} ({result['year']})")
    print(f"  Pages: {result['total_pages']}")
    print(f"  Elements: {result['total_elements']}")
    print(f"  Quality: {result['quality_composite']:.3f} "
          f"(pass rate: {result['quality_pass_rate']:.1%})")
    print(f"  Time: {result['processing_time_seconds']:.1f}s")
    print(f"  Output: {result['output_path']}")


def cmd_batch(args):
    """Batch process multiple PDF documents."""
    from PRISM.config import PRISMConfig
    from PRISM.pipeline import PRISMPipeline
    from PRISM.visualization.export import DataExporter

    config = PRISMConfig(domain=args.domain)
    if args.output_dir:
        config.output_dir = args.output_dir

    # Discover PDFs
    input_dir = Path(args.input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {args.input_dir}")
        sys.exit(1)

    documents = []
    for pdf in pdf_files:
        country, year = DataExporter.extract_country_year(pdf.stem + ".json")
        documents.append({
            "pdf_path": str(pdf),
            "country": country,
            "year": year,
        })

    print(f"Found {len(documents)} PDF files")
    pipeline = PRISMPipeline(config)
    results = pipeline.process_batch(
        documents, lightweight_ingestion=args.lightweight
    )

    successful = sum(1 for r in results if "error" not in r)
    print(f"\nBatch complete: {successful}/{len(results)} successful")


def cmd_validate(args):
    """Validate existing analysis JSON files."""
    from PRISM.validation import AnalysisValidator

    validator = AnalysisValidator(domain=args.domain)
    results = validator.validate_folder(args.folder, verbose=not args.quiet)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")


def cmd_export(args):
    """Export JSON analyses to CSV."""
    from PRISM.visualization.export import DataExporter

    if args.commonwealth:
        from PRISM.validation import COMMONWEALTH_COUNTRIES
        global_rows, cw_rows = DataExporter.export_global_and_commonwealth(
            args.folder, args.output, COMMONWEALTH_COUNTRIES
        )
        print(f"Global CSV: {global_rows} rows")
        print(f"Commonwealth CSV: {cw_rows} rows")
    else:
        output_path = str(Path(args.output) / "analyses.csv")
        rows = DataExporter.export_folder_to_csv(args.folder, output_path)
        print(f"CSV exported: {rows} rows")


def cmd_summary(args):
    """Generate summary statistics for analyzed plans."""
    from PRISM.visualization.scores import ScoreAnalyzer

    analyzer = ScoreAnalyzer(domain=args.domain)
    stats = analyzer.generate_summary_statistics(args.folder)

    print(f"\nPRISM Summary Statistics ({args.domain.upper()})")
    print("=" * 50)
    print(f"Total plans analyzed: {stats.get('total_plans', 0)}")
    print(f"Median overall score: {stats.get('median_overall', 0)}/5")
    print(f"Mean overall score:   {stats.get('mean_overall', 0)}/5")

    print(f"\nHighest scoring sections:")
    for s in stats.get("highest_sections", []):
        print(f"  {s['section']}: {s['median']}")

    print(f"\nLowest scoring sections:")
    for s in stats.get("lowest_sections", []):
        print(f"  {s['section']}: {s['median']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nFull statistics saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="prism",
        description=(
            "PRISM - Policy Reasoning Integrated Sequential Model\n"
            "Multi-Agent AI System for National Disease Control Plan Analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── analyze ──
    p_analyze = subparsers.add_parser("analyze", help="Analyze a single document")
    p_analyze.add_argument("--pdf", required=True, help="Path to PDF document")
    p_analyze.add_argument("--country", required=True, help="Country name")
    p_analyze.add_argument("--year", required=True, help="Plan publication year")
    p_analyze.add_argument("--domain", default="cancer",
                           choices=["cancer", "cvd"], help="Analysis domain")
    p_analyze.add_argument("--output", help="Output JSON path")
    p_analyze.add_argument("--output-dir", help="Output directory")
    p_analyze.add_argument("--config", help="Path to config JSON")
    p_analyze.add_argument("--lightweight", action="store_true",
                           help="Use lightweight ingestion (no VL model)")

    # ── batch ──
    p_batch = subparsers.add_parser("batch", help="Batch process PDF folder")
    p_batch.add_argument("--input-dir", required=True, help="Folder with PDFs")
    p_batch.add_argument("--domain", default="cancer",
                         choices=["cancer", "cvd"])
    p_batch.add_argument("--output-dir", help="Output directory")
    p_batch.add_argument("--lightweight", action="store_true")

    # ── validate ──
    p_validate = subparsers.add_parser("validate", help="Validate analysis JSONs")
    p_validate.add_argument("--folder", required=True,
                            help="Folder with JSON analyses")
    p_validate.add_argument("--domain", default="cancer",
                            choices=["cancer", "cvd"])
    p_validate.add_argument("--output", help="Save report to JSON")
    p_validate.add_argument("--quiet", "-q", action="store_true")

    # ── export ──
    p_export = subparsers.add_parser("export", help="Export analyses to CSV")
    p_export.add_argument("--folder", required=True,
                          help="Folder with JSON analyses")
    p_export.add_argument("--output", required=True, help="Output directory")
    p_export.add_argument("--commonwealth", action="store_true",
                          help="Also generate Commonwealth CSV")

    # ── summary ──
    p_summary = subparsers.add_parser("summary", help="Generate summary stats")
    p_summary.add_argument("--folder", required=True,
                           help="Folder with JSON analyses")
    p_summary.add_argument("--domain", default="cancer",
                           choices=["cancer", "cvd"])
    p_summary.add_argument("--output", help="Save stats to JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "analyze": cmd_analyze,
        "batch": cmd_batch,
        "validate": cmd_validate,
        "export": cmd_export,
        "summary": cmd_summary,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
