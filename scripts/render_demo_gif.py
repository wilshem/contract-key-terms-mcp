"""Draw docs/demo.gif from scripts/demo.py's real output, frame by frame, with Pillow (no screen recording).

Usage: uv run --group demo python scripts/render_demo_gif.py [--out docs/demo.gif]
"""
import argparse
import asyncio
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import transcript

COLS, ROWS = 104, 26
FONT_SIZE = 15
PAD = 16
BG, FG, CMD, DIM = (24, 26, 33), (220, 223, 228), (126, 200, 129), (140, 146, 158)
FONTS = ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "Menlo.ttc", "Consolas.ttf"]
TYPE_MS, LINE_MS = 35, 90


def font() -> ImageFont.FreeTypeFont:
    for name in FONTS:
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    raise SystemExit("no monospace font found (tried DejaVu Sans Mono, Menlo, Consolas)")


class Screen:
    def __init__(self):
        self.f = font()
        box = self.f.getbbox("M")
        self.cw, self.ch = self.f.getlength("M"), int((box[3] - box[1]) * 1.45)
        self.size = (int(self.cw * COLS) + 2 * PAD, self.ch * ROWS + 2 * PAD)
        self.lines: list[tuple[str, tuple]] = []
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def frame(self, ms: int) -> None:
        img = Image.new("RGB", self.size, BG)
        d = ImageDraw.Draw(img)
        for row, (text, color) in enumerate(self.lines[-ROWS:]):
            d.text((PAD, PAD + row * self.ch), text[:COLS], font=self.f, fill=color)
        self.frames.append(img)
        self.durations.append(ms)

    def type_command(self, text: str) -> None:
        self.lines.append(("$ ", CMD))
        for i in range(0, len(text), 2):
            self.lines[-1] = ("$ " + text[: i + 2], CMD)
            self.frame(TYPE_MS)
        self.frame(300)

    def output(self, text: str) -> None:
        self.lines.append((text, DIM if text.strip().startswith('"') else FG))
        self.frame(LINE_MS)

    def pause(self, seconds: float) -> None:
        self.durations[-1] += int(seconds * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "demo.gif"))
    args = ap.parse_args()
    s = Screen()
    s.frame(500)
    for kind, value in asyncio.run(transcript()):
        if kind == "cmd":
            if len(s.lines) > 1:
                s.lines.append(("", FG))
            s.type_command(str(value))
        elif kind == "out":
            s.output(str(value))
        else:
            s.pause(float(value))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.frames[0].save(out, save_all=True, append_images=s.frames[1:], duration=s.durations, loop=0, optimize=True)
    print(f"{out}: {len(s.frames)} frames, {sum(s.durations) / 1000:.1f} s, {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
