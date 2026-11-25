"""Export TODO.md to TODO.xlsx.

TODO.md is the source of truth. Run this script after editing it:

    .venv\\Scripts\\python.exe scripts\\export_todo.py

Optional inline tags inside a task line:
    - [ ] Tugas [P: Tinggi] [D: 2026-11-21] [N: catatan singkat]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "TODO.md"
TARGET = ROOT / "TODO.xlsx"

STATUS_ORDER = ["Selesai", "Sedang dikerjakan", "Berikutnya", "Ide"]
DEFAULT_PRIORITY = {
    "Sedang dikerjakan": "Tinggi",
    "Berikutnya": "Sedang",
    "Ide": "Rendah",
}
STATUS_STYLE = {
    "Selesai": ("C6EFCE", "006100"),
    "Sedang dikerjakan": ("FFEB9C", "9C6500"),
    "Berikutnya": ("DDEBF7", "1F4E79"),
    "Ide": ("D9D9D9", "404040"),
}
PRIORITY_COLOR = {"Tinggi": "C00000", "Sedang": "BF8F00", "Rendah": "375623"}

TAG = re.compile(r"\[(P|N|D):\s*([^\]]+)\]")
TASK = re.compile(r"^[-*]\s+\[[ xX]\]\s+(.*)$")


@dataclass
class Todo:
    task: str
    status: str
    priority: str
    updated: str
    note: str


def section_status(heading: str) -> str | None:
    for status in STATUS_ORDER:
        if status.lower() in heading.lower():
            return status
    return None


def parse(path: Path) -> list[Todo]:
    todos: list[Todo] = []
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = section_status(stripped[3:])
            continue
        if current is None:
            continue
        match = TASK.match(stripped)
        if not match:
            continue
        text = match.group(1).strip()
        tags = {key: value.strip() for key, value in TAG.findall(text)}
        text = TAG.sub("", text).strip().replace("`", "")
        if not text:
            continue
        todos.append(
            Todo(
                task=text,
                status=current,
                priority=tags.get("P", DEFAULT_PRIORITY.get(current, "—")),
                updated=tags.get("D", date.today().isoformat()),
                note=tags.get("N", "—"),
            )
        )
    order = {status: index for index, status in enumerate(STATUS_ORDER)}
    todos.sort(key=lambda item: order[item.status])
    return todos


def fill(color: str) -> PatternFill:
    return PatternFill("solid", start_color=color, end_color=color)


def build(todos: list[Todo], sheet: Worksheet) -> None:
    headers = ["No", "Tugas", "Status", "Prioritas", "Tanggal", "Catatan"]
    widths = [6, 62, 20, 12, 14, 38]
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    sheet.append(headers)
    for column, width in enumerate(widths, start=1):
        cell = sheet.cell(row=1, column=column)
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.fill = fill("1F4E79")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border
        sheet.column_dimensions[cell.column_letter].width = width
    sheet.row_dimensions[1].height = 24

    for number, todo in enumerate(todos, start=1):
        sheet.append([number, todo.task, todo.status, todo.priority, todo.updated, todo.note])
        row = sheet.max_row
        background, foreground = STATUS_STYLE.get(todo.status, ("FFFFFF", "000000"))
        for column in range(1, len(headers) + 1):
            cell = sheet.cell(row=row, column=column)
            cell.border = border
            cell.alignment = Alignment(
                horizontal="center" if column in (1, 3, 4, 5) else "left",
                vertical="top",
                wrap_text=column in (2, 6),
            )
            if column == 3:
                cell.fill = fill(background)
                cell.font = Font(color=foreground, bold=True)
            elif column == 4:
                cell.font = Font(color=PRIORITY_COLOR.get(todo.priority, "000000"))
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:F{sheet.max_row}"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}")
    todos = parse(SOURCE)
    workbook = Workbook()
    build(todos, workbook.active)
    workbook.active.title = "TODO"
    workbook.save(TARGET)
    print(f"Wrote {TARGET} ({len(todos)} tasks)")


if __name__ == "__main__":
    main()
