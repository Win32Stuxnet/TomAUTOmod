from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "not-a-surveillance-tool-summary.pdf"


TITLE = "Not-a-Surveillance-Tool"

WHAT_IT_IS = (
    "A work-in-progress Discord moderation bot plus FastAPI dashboard for managing "
    "server safety features. The codebase combines rule-based automod, moderator "
    "workflows, and optional ML-assisted review and training backed by MongoDB."
)

WHO_ITS_FOR = (
    "Discord server admins and moderators who want configurable moderation, review, "
    "logging, ticketing, and lightweight dashboard controls."
)

FEATURES = [
    "Runs a Discord bot and web dashboard together in one app process.",
    "Provides slash-command moderation actions, bulk purge, and case tracking.",
    "Applies word, regex, and link filters plus anti-spam and raid protection.",
    "Supports ticket intake and ticket claim, add, remove, and close flows.",
    "Offers a Discord OAuth dashboard to edit guild config, filters, and custom commands.",
    "Stores guild config, cases, tickets, cached messages, custom commands, and ML data in MongoDB.",
    "Collects and labels message samples for ML review and can train and load a local model file.",
]

ARCHITECTURE = [
    "main.py creates MongoDB indexes, then starts FastAPI and ModBot together with asyncio.gather.",
    "bot/bot.py loads all cogs, connects to MongoDB through bot/database.py, and initializes the ML collector.",
    "Discord commands and events flow through bot/cogs and bot/services, which read and write MongoDB collections.",
    "FastAPI serves templates and static assets, handles Discord OAuth sessions, and exposes API endpoints used by the dashboard JavaScript.",
    "ML flow in repo: message -> feature extraction and prediction -> audit or review UI -> labeled samples -> optional training to model.joblib.",
]

RUN_STEPS = [
    "From the repo root, create env vars for `DISCORD_TOKEN`, MongoDB settings, and dashboard OAuth settings. Exact sample `.env` values: Not found in repo.",
    "Start with `docker compose up --build` to launch the app and MongoDB together.",
    "Alternative local path: install `requirements.txt`, ensure MongoDB is running, then run `python main.py`.",
    "Open `http://localhost:8080` for the dashboard. Discord login requires `DISCORD_CLIENT_ID` and `DISCORD_CLIENT_SECRET`.",
    "Production deployment guidance and a full onboarding guide: Not found in repo.",
]


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, font: str, size: int, leading: float):
    c.setFont(font, size)
    lines = simpleSplit(text, font, size, width)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_bullets(c: canvas.Canvas, items: list[str], x: float, y: float, width: float, font: str, size: int, leading: float):
    bullet_x = x
    text_x = x + 10
    wrap_width = width - 10
    for item in items:
        lines = simpleSplit(item, font, size, wrap_width)
        c.setFont(font, size)
        c.drawString(bullet_x, y, "-")
        for idx, line in enumerate(lines):
            c.drawString(text_x, y, line)
            if idx < len(lines) - 1:
                y -= leading
        y -= leading + 1
    return y


def section_heading(c: canvas.Canvas, label: str, x: float, y: float):
    c.setFillColor(colors.HexColor("#123d5a"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, label)
    c.setFillColor(colors.black)
    return y - 14


def build_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    left = 40
    right = width - 40
    text_width = right - left
    y = height - 42

    c.setTitle(TITLE)
    c.setStrokeColor(colors.HexColor("#b8c7d3"))
    c.setFillColor(colors.HexColor("#0c2436"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(left, y, TITLE)
    c.setFillColor(colors.HexColor("#52616b"))
    c.setFont("Helvetica", 9)
    c.drawRightString(right, y + 1, "Repo summary - one page")
    y -= 14
    c.line(left, y, right, y)
    y -= 16

    y = section_heading(c, "What It Is", left, y)
    y = draw_wrapped(c, WHAT_IT_IS, left, y, text_width, "Helvetica", 9, 11)
    y -= 4

    y = section_heading(c, "Who It's For", left, y)
    y = draw_wrapped(c, WHO_ITS_FOR, left, y, text_width, "Helvetica", 9, 11)
    y -= 4

    y = section_heading(c, "What It Does", left, y)
    y = draw_bullets(c, FEATURES, left + 2, y, text_width - 2, "Helvetica", 9, 11)
    y -= 2

    y = section_heading(c, "How It Works", left, y)
    y = draw_bullets(c, ARCHITECTURE, left + 2, y, text_width - 2, "Helvetica", 9, 11)
    y -= 2

    y = section_heading(c, "How To Run", left, y)
    y = draw_bullets(c, RUN_STEPS, left + 2, y, text_width - 2, "Helvetica", 9, 11)

    if y < 36:
        raise RuntimeError("Content overflowed the single page layout.")

    c.setStrokeColor(colors.HexColor("#d7e0e7"))
    c.line(left, 30, right, 30)
    c.setFillColor(colors.HexColor("#52616b"))
    c.setFont("Helvetica", 7)
    c.drawString(left, 20, "Based on repo evidence inspected on 2026-03-12.")
    c.drawRightString(right, 20, "Missing setup details are marked explicitly.")

    c.save()


if __name__ == "__main__":
    build_pdf(OUTPUT)
    print(OUTPUT)
