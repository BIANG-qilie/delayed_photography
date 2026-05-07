# delayed-photography

从视频合成单张「堆栈 / 延时」风格 RGB 照片（Python 库）。使用 **PyAV** 解码，**NumPy** 做像素级合成。

## 安装

```bash
pip install delayed-photography
```

开发环境（可编辑安装 + 工具链）：

```bash
pip install -e ".[dev]"
```

**依赖说明**：需要已安装带 FFmpeg 的 **PyAV**（`av`）。在 Windows 上通常可直接 `pip install av`（有预编译 wheel）；若失败，请检查 Python 版本（建议 3.10+）与网络/代理。

## 快速用法

```python
from pathlib import Path
from delayed_photography import compose

# 长曝光感：全帧均值
img = compose(Path("input.mp4"), mode="mean", sampling="all")

# 光轨：每隔 3 帧取一帧做逐像素最大值
img = compose("input.mp4", mode="max", sampling="step", sampling_param=3)

# 去游客：均匀抽 120 帧再算中位数（强烈建议 median 限制帧数）
img = compose(
    "input.mp4",
    mode="median",
    sampling="count",
    sampling_param=120,
    output_path="out.png",
)
```

`compose()` 返回 `numpy.ndarray`，`dtype=uint8`，形状 `(H, W, 3)`（RGB）。若传入 `output_path`，会同时写入 `.png` / `.jpg` / `.jpeg` 文件。

## 模式说明

| `mode`   | 效果简述 | 内存特点 |
|----------|----------|----------|
| `mean`   | 逐帧平均，类似长曝光、拉丝水流/人群 | 流式 O(1) 帧缓冲 |
| `max`    | 逐通道取最大值，适合车灯轨迹、烟花 | 流式 O(1) 帧缓冲 |
| `median` | 中位数去运动物体（「空场景」） | 需缓存**全部参与合成的帧**，大视频请配合 `sampling` |

## 抽帧 `sampling`

- `all`：使用每一帧（`sampling_param` 忽略）。
- `step`：每 `sampling_param` 帧取 1 帧（从第 0 帧开始）。
- `count`：在整段视频中**均匀**抽取 `sampling_param` 帧（含首尾，去重排序）。

## 中位数与内存

`median` 会为参与合成的每一帧保存一份拷贝。超过 `max_median_bytes`（默认 512 MiB）的**预估**占用会触发 `MemoryError`。长视频请优先使用 `sampling="count"` 或 `"step"`。

## 示例脚本

将 `d779888b8a8f9962ca01307547cd36cf.mp4` 放到 `example/data/` 后运行：

```bash
PYTHONPATH=src python example/scripts/extract_in_video/demo_compose_params.py
```

PowerShell：

```powershell
$env:PYTHONPATH = "src"; python example/scripts/extract_in_video/demo_compose_params.py
```

输出写入 `example/output/`（该目录已在 `.gitignore` 中忽略）。

## 开发与校验

```bash
ruff check src tests
mypy src
pytest tests -v
python -m build
twine check dist/*
```

## 许可证

MIT
