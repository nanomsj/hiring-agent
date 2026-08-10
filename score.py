import os
import sys
import json

# Fix for Windows Console Unicode errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Fix for Python 3.14 Protobuf TypeError
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import csv

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import argparse
import datetime

from pdf import PDFHandler
from models import JSONResume, build_evaluation_model
from evaluator import ResumeEvaluator
from roles import Role, load_role, list_available_roles, scaffold_role
from jd_role import generate_role_from_jd
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from transform import (
    transform_evaluation_response,
    convert_json_resume_to_text,
    convert_blog_data_to_text,
    CATEGORY_CN,
)
from config import DEVELOPMENT_MODE

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)


def print_evaluation_results(
    evaluation, role: Role, candidate_name: str = "Candidate"
):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print(f"📊 RESUME EVALUATION RESULTS FOR: {candidate_name}")
    print("=" * 80)

    if not evaluation:
        print("❌ No evaluation data available")
        return None

    # Calculate overall score
    total_score = 0
    max_score = 0
    category_scores = {}

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            category_score = min(category_data["score"], category_data["max"])
            total_score += category_score
            max_score += category_data["max"]
            category_scores[category_name] = (category_score, category_data["max"])

            # Log warning if score was capped
            if category_score < category_data["score"]:
                print(
                    f"⚠️  Warning: {category_name} score capped from {category_data['score']} to {category_score} (max: {category_data['max']})"
                )

    # Add bonus points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    # Subtract deductions
    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= evaluation.deductions.total

    # Ensure total score doesn't exceed maximum possible score
    max_possible_score = max_score + role.bonus_max
    if total_score > max_possible_score:
        total_score = max_possible_score
        print(f"⚠️  Warning: Total score capped at maximum possible value")

    # Overall Score
    print(f"\n🎯 OVERALL SCORE: {total_score:.1f}/{max_score}")

    # Detailed Scores
    print("\n📈 DETAILED SCORES:")
    print("-" * 60)

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category in role.categories:
            cat_score = getattr(evaluation.scores, category.key, None)
            if not cat_score:
                continue
            capped_score = min(cat_score.score, category.max)
            print(f"{category.icon} {category.label}: {capped_score}/{cat_score.max}")
            print(f"   Evidence: {cat_score.evidence}")
            print()

    # Bonus Points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        print(f"\n⭐ BONUS POINTS: {evaluation.bonus_points.total}")
        print("-" * 30)
        print(f"   {evaluation.bonus_points.breakdown}")

    # Deductions
    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and evaluation.deductions.total > 0
    ):
        print(f"\n⚠️  DEDUCTIONS: -{evaluation.deductions.total}")
        print("-" * 30)
        if evaluation.deductions.reasons:
            print(f"   {evaluation.deductions.reasons}")

    # Key Strengths
    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        print(f"\n✅ KEY STRENGTHS:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    # Areas for Improvement
    if (
        hasattr(evaluation, "areas_for_improvement")
        and evaluation.areas_for_improvement
    ):
        print(f"\n🔧 AREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    print("\n" + "=" * 80)

    return {
        "total_score": total_score,
        "total_max": max_score,
        "bonus": (
            evaluation.bonus_points.total
            if hasattr(evaluation, "bonus_points") and evaluation.bonus_points
            else 0
        ),
        "deductions": (
            evaluation.deductions.total
            if hasattr(evaluation, "deductions") and evaluation.deductions
            else 0
        ),
        "categories": category_scores,
    }


def _display_width(s: str) -> int:
    """Approximate terminal column width, counting CJK characters as 2 columns."""
    return sum(2 if ord(ch) > 127 else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    """Right-pad a string to a display width, honoring CJK double-width chars."""
    return s + " " * max(0, width - _display_width(s))


def print_summary_table(summaries: list, role: Role):
    """Print a compact overall summary table after a batch run."""
    if not summaries:
        print("\n❌ 没有可汇总的结果")
        return

    label_by_key = {
        c.key: CATEGORY_CN.get(c.label, c.label) for c in role.categories
    }
    headers = ["文件名", "姓名", "总分", "满分"]
    headers += [label_by_key[c.key] for c in role.categories]
    headers += ["加分", "扣分"]

    def row_cells(summary):
        cells = [
            str(summary.get("file_name", "")),
            str(summary.get("name", "")),
            f"{summary.get('total_score', 0):.1f}",
            str(summary.get("total_max", 0)),
        ]
        for c in role.categories:
            pair = summary.get("categories", {}).get(c.key)
            cells.append(f"{pair[0]:.1f}/{pair[1]}" if pair else "N/A")
        cells.append(str(summary.get("bonus", 0)))
        cells.append(str(summary.get("deductions", 0)))
        return cells

    rows = [row_cells(s) for s in summaries]

    # Column widths = widest cell (headers included).
    widths = [_display_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _display_width(cell))

    table_width = sum(widths) + 4 * len(headers)
    separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells):
        return "| " + " | ".join(_pad(c, widths[i]) for i, c in enumerate(cells)) + " |"

    print("\n" + "=" * table_width)
    print("📊 批量处理汇总表")
    print(separator)
    print(fmt_row(headers))
    print(separator)
    for row in rows:
        print(fmt_row(row))
    print(separator)


def _evaluate_resume(
    resume_data: JSONResume,
    role: Role,
    evaluation_model,
    blog_data: dict = None,
):
    """Evaluate the resume using AI and display results."""

    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(
        role=role,
        evaluation_model=evaluation_model,
        model_name=DEFAULT_MODEL,
        model_params=model_params,
    )

    # Convert JSON resume data to text
    resume_text = convert_json_resume_to_text(resume_data)

    # Add blog data if available
    if blog_data:
        blog_text = convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    # Evaluate the enhanced resume
    evaluation_result = evaluator.evaluate_resume(resume_text)

    # print(evaluation_result)

    return evaluation_result


def is_valid_resume_data(resume_data: JSONResume) -> bool:
    """Check if the resume data has at least some extracted core content."""
    if not resume_data:
        return False
    core_sections = [
        resume_data.basics,
        resume_data.work,
        resume_data.education,
        resume_data.skills,
        resume_data.projects,
    ]
    return any(section is not None for section in core_sections)


def main(pdf_path, role: Role):
    print(f"📄 正在处理：{os.path.basename(pdf_path)}")
    evaluation_model = build_evaluation_model(role)

    # Create cache filename based on PDF path
    cache_filename = (
        f"cache/resumecache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    )

    resume_data = None
    cache_loaded = False

    # Check if cache exists and we're in development mode
    if DEVELOPMENT_MODE and os.path.exists(cache_filename):
        print("   ✅ 命中缓存，直接使用已解析的简历数据")
        try:
            cached_data = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
            loaded_resume = JSONResume(**cached_data)
            if not is_valid_resume_data(loaded_resume):
                raise ValueError("Cached resume data contains no core content")
            resume_data = loaded_resume
            cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid cache file {cache_filename}: {e}")
            print("Ignoring cache and reprocessing PDF...")
            try:
                os.remove(cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid cache file {cache_filename}: {delete_err}"
                )

    if not cache_loaded:
        logger.debug(
            f"Extracting data from PDF"
            + (" and caching to " + cache_filename if DEVELOPMENT_MODE else "")
        )
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)

        if resume_data == None:
            return None, None

        if DEVELOPMENT_MODE:
            if is_valid_resume_data(resume_data):
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                Path(cache_filename).write_text(
                    json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                logger.warning(
                    "Newly extracted resume data is empty/invalid. Skipping cache write."
                )

    score = _evaluate_resume(resume_data, role, evaluation_model)

    # Get candidate name for display
    candidate_name = os.path.basename(pdf_path).replace(".pdf", "")
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    # Print evaluation results in readable format, capturing a compact summary
    summary = print_evaluation_results(score, role, candidate_name)
    if summary:
        summary["file_name"] = os.path.basename(pdf_path)
        summary["name"] = candidate_name

    # Build the CSV row (written to the results file by run_scoring).
    csv_row = None
    if DEVELOPMENT_MODE:
        csv_row = transform_evaluation_response(
            file_name=os.path.basename(pdf_path),
            evaluation=score,
            resume_data=resume_data,
            role=role,
        )

    return summary, csv_row


def save_results_csv(role, rows, output_dir=None) -> Path | None:
    """Write all scoring rows to one timestamped CSV under the result directory.

    The file is named ``<role>_<YYYYMMDD_HHMMSS>.csv``. The directory defaults to
    the sibling of this project (``xingbodongli/results``) so results never clutter
    the source tree, unless an explicit ``output_dir`` is given.
    """
    if not rows:
        print("   ⚠️  没有可写的评分结果行，跳过 CSV 输出。")
        return None

    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Default: <project parent>/results  (sibling of hiring-agent/).
        out_dir = Path(__file__).resolve().parent.parent / "results"

    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{role.name}_{timestamp}.csv"

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("   ✅ 评分结果已保存：")
    print(f"   {csv_path.resolve()}")
    return csv_path


def run_role_generation(args) -> None:
    """`--jd` flow ONLY: build a role from a job description, print it, exit.

    Never reads, scans or scores any resume.
    """
    generate_role_from_jd(args.jd, role_key=args.role_key, force=args.force)


def run_scoring(args, role: Role) -> None:
    """Score one PDF or a folder of PDFs against an EXISTING role.

    This function never creates, modifies or overwrites any role — it only
    loads the role passed in and scores resumes against it. All CSV rows are
    collected and written once to a timestamped file in the result directory.
    """

    rows = []
    summaries = []

    # Batch mode: pdf_path is a directory — process every .pdf inside it.
    if os.path.isdir(args.pdf_path):
        pdf_files = sorted(Path(args.pdf_path).glob("*.pdf"))
        if not pdf_files:
            print(f"❌ 目录中没有 PDF 文件：{args.pdf_path}")
            exit(1)
        print(f"📂 找到 {len(pdf_files)} 份简历，开始批量处理...")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'=' * 70}")
            print(f"📄 进度（{i}/{len(pdf_files)}）：{pdf_file.name}")
            summary, csv_row = main(str(pdf_file), role)
            if summary:
                summaries.append(summary)
            if csv_row:
                rows.append(csv_row)
        print_summary_table(summaries, role)
    else:
        summary, csv_row = main(args.pdf_path, role)
        if summary:
            summaries.append(summary)
        if csv_row:
            rows.append(csv_row)

    save_results_csv(role, rows, getattr(args, "output_dir", None))


if __name__ == "__main__":
    available_roles = list_available_roles()
    parser = argparse.ArgumentParser(
        description="Generate a scoring role from a JD, or score a resume against an existing role."
    )
    parser.add_argument(
        "pdf_path", nargs="?", help="Path to the resume PDF to evaluate"
    )
    parser.add_argument(
        "--role",
        help="Role to score against (a directory name under roles/). "
        + (f"Available: {', '.join(available_roles)}" if available_roles else ""),
    )
    parser.add_argument(
        "--jd",
        metavar="JD_FILE",
        help="Path to a natural-language job description text file. Generates a "
        "new role under roles/ and exits WITHOUT scoring any resume. Mutually "
        "exclusive with pdf_path and --role.",
    )
    parser.add_argument(
        "--role-key",
        metavar="NAME",
        help="Optional safe name for the role generated from --jd (e.g. "
        "mechanical_design_intern -> roles/mechanical_design_intern/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --jd: regenerate the role even if the target directory "
        "already exists.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory for result CSVs. Defaults to ../results (a sibling "
        "of this project). The directory is created if missing.",
    )
    parser.add_argument(
        "--init-role",
        metavar="NAME",
        help="Scaffold a new role directory under roles/ with basic template "
        "files, then exit (does not score a resume).",
    )
    args = parser.parse_args()

    # Scaffold mode: create a new role and exit.
    if args.init_role:
        try:
            role_dir = scaffold_role(args.init_role)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
        print(f"✅ Created role '{args.init_role}' at {role_dir}")
        print("   Edit role.json, criteria.jinja and system_message.jinja, then run:")
        print(f"   python score.py <pdf_path> --role {args.init_role}")
        exit(0)

    # Step 1 — role generation from a JD. Exclusive; no pdf_path allowed.
    if args.jd:
        if args.pdf_path:
            parser.error("conflicting arguments: pdf_path and --jd cannot be used together")
        if args.role:
            parser.error("conflicting arguments: --role and --jd cannot be used together")
        try:
            run_role_generation(args)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}")
            exit(1)
        exit(0)

    # Step 2 — resume scoring against an already-existing role.
    if not args.pdf_path or not args.role:
        parser.error("pdf_path and --role are required (or use --jd to generate a role)")

    if not os.path.exists(args.pdf_path):
        print(f"Error: '{args.pdf_path}' does not exist.")
        exit(1)

    try:
        role = load_role(args.role)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    run_scoring(args, role)
