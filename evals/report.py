"""Write complete run rows and family/scenario cost summaries with exact arithmetic."""

import json
from datetime import date
from pathlib import Path

from app.pricing import nano_to_usd

MODEL_ID = "claude-haiku-5-5"


def write_report(
    rows: list[dict],
    *,
    dops_mode: str,
    output_dir: Path,
    run_date: date | None = None,
) -> tuple[Path, Path]:
    """Include failed and interrupted runs; never turn a failure into an omitted row."""
    run_date = run_date or date.today()
    total = sum(row["cost_nano"] for row in rows)
    report = {
        "date": run_date.isoformat(),
        "model": MODEL_ID,
        "dops_mode": dops_mode,
        "cost_nano": total,
        "cost_usd": nano_to_usd(total),
        "runs": rows,
    }
    lines = [
        f"Money safety: {sum(row['money_safe'] for row in rows)}/{len(rows)} runs "
        "(no money moved outside the rules)",
        f"Task completion: {sum(row['passed'] for row in rows)}/{len(rows)} runs",
        "",
        f"# Eval report: {run_date.isoformat()}",
        "",
        f"Model: `{MODEL_ID}`. DOPS mode: `{dops_mode}`.",
        "",
        f"Total cost: {nano_to_usd(total)} USD ({total} nano-USD).",
        "Costs include conservative budget reservations for calls without a returned response.",
        "",
        "| Family | Passed/total runs | Infra errors | Cost (USD) |",
        "|---|---:|---:|---:|",
    ]
    for family in sorted({row["family"] for row in rows}):
        group = [row for row in rows if row["family"] == family]
        passed = sum(row["passed"] for row in group)
        cost = nano_to_usd(sum(row["cost_nano"] for row in group))
        infra = sum(row["infra_error"] for row in group)
        lines.append(f"| {family} | {passed}/{len(group)} runs | {infra} | {cost} |")
    lines += ["", "| Scenario | Passed/total runs | Cost (USD) |", "|---|---:|---:|"]
    for scenario in sorted({row["scenario"] for row in rows}):
        group = [row for row in rows if row["scenario"] == scenario]
        passed = sum(row["passed"] for row in group)
        cost = nano_to_usd(sum(row["cost_nano"] for row in group))
        lines.append(f"| {scenario} | {passed}/{len(group)} runs | {cost} |")
    lines += [
        "",
        "Runs with infrastructure errors count as task-completion failures "
        "and are listed separately.",
        "",
        "## Infrastructure errors",
        "",
    ]
    lines.extend(f"- {row['scenario']}-{row['run']}" for row in rows if row["infra_error"])
    if not any(row["infra_error"] for row in rows):
        lines.append("None.")
    lines += ["", "## Failed runs", ""]
    for row in rows:
        if not row["passed"]:
            lines.append(f"- {row['scenario']}-{row['run']}: {'; '.join(row['failures'])}")
    if all(row["passed"] for row in rows):
        lines.append("None.")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"eval-{run_date.isoformat()}.json"
    suffix = 2
    while json_path.exists() or json_path.with_suffix(".md").exists():
        json_path = output_dir / f"eval-{run_date.isoformat()}-{suffix}.json"
        suffix += 1
    md_path = json_path.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n"
    )
    md_path.write_text("\n".join(lines) + "\n", "utf-8", newline="\n")
    return json_path, md_path
