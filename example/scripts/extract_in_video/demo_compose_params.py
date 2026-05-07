"""Demonstrate delayed_photography.compose() parameters on a local MP4.

Input (expected, not tracked by git):
  example/data/testvideo.mp4

Outputs:
  example/output/{video_stem}__{mode}__{sampling}{param}.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from delayed_photography import compose

Mode = Literal["mean", "max", "median"]
Sampling = Literal["all", "step", "count"]

# median 会缓存所有参与合成的帧；step=5 在长视频上帧数≈N/5，512MiB 很容易不够。
MEDIAN_DEMO_MAX_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class DemoCase:
    mode: Mode
    sampling: Sampling
    sampling_param: int | None = None
    max_median_bytes: int | None = None

    def out_name(self, video_stem: str) -> str:
        if self.sampling == "all":
            sampling_part = "all"
        else:
            sampling_part = f"{self.sampling}{self.sampling_param}"
        return f"{video_stem}__{self.mode}__{sampling_part}.png"


def repo_root() -> Path:
    # example/scripts/extract_in_video/demo_compose_params.py -> repo root is 4 parents up
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = repo_root()
    video_path = root / "example" / "data" / "testvideo.mp4"
    out_dir = root / "example" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        msg = (
            "未找到输入视频文件。\n"
            f"请把 MP4 放到：{video_path}\n"
            "注意：仓库 .gitignore 默认忽略 *.mp4（示例数据不应提交到 git）。"
        )
        print(msg, file=sys.stderr)
        return 2

    video_stem = video_path.stem

    cases: list[DemoCase] = [
        DemoCase(mode="mean", sampling="all"),
        DemoCase(mode="mean", sampling="step", sampling_param=3),
        DemoCase(mode="mean", sampling="count", sampling_param=120),
        DemoCase(mode="max", sampling="all"),
        DemoCase(mode="max", sampling="step", sampling_param=3),
        DemoCase(mode="max", sampling="count", sampling_param=120),
        DemoCase(
            mode="median",
            sampling="count",
            sampling_param=120,
            max_median_bytes=MEDIAN_DEMO_MAX_BYTES,
        ),
        DemoCase(
            mode="median",
            sampling="step",
            sampling_param=5,
            max_median_bytes=MEDIAN_DEMO_MAX_BYTES,
        ),
    ]

    for c in cases:
        out_path = out_dir / c.out_name(video_stem)
        print(
            f"[run] mode={c.mode} sampling={c.sampling} param={c.sampling_param} -> {out_path}"
        )
        kwargs: dict[str, object] = {
            "mode": c.mode,
            "sampling": c.sampling,
            "output_path": out_path,
        }
        if c.sampling != "all":
            kwargs["sampling_param"] = int(c.sampling_param or 1)
        if c.max_median_bytes is not None:
            kwargs["max_median_bytes"] = int(c.max_median_bytes)
        compose(video_path, **kwargs)  # type: ignore[arg-type]

    print(f"[done] outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
