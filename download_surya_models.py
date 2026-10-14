"""把 Surya（含 surya-ocr-2 VLM）所需的模型下载到一个本地目录。

设计目标（Guaxi 专用，等价于上游 `surya.common.s3` 的下载逻辑，但可以脱离 surya/torch
运行，便于在 ML 环境就绪之前先把模型准备好）：

1. **不 import surya**：早期版本需要先有完整依赖才能下模型，一旦 torch/cv2 装不上就彻底
   卡死。本脚本只用标准库 + huggingface_hub，因此可以独立运行。
2. **断点续传**：一律先写 `<name>.part`，支持 `Range` 续传，中断后重跑不会从头再来。
3. **跟随代理**：走 `urllib` / `requests` 默认行为——读 `HTTP(S)_PROXY` 等环境变量，
   在 Windows 上还会读系统代理设置；没有代理就直连。`HF_ENDPOINT` 可用于换 HF 镜像。
4. **多源兜底**：`s3://` 走 `models.datalab.to`（境内直连通常很快），HF 权重走
   `HF_ENDPOINT`（默认官方，可设成 https://hf-mirror.com）。

模型集合按 surya 0.22.x 的 checkpoint 定义：

- `s3://text_detection/2025_05_07`            -> MODEL_CACHE_DIR/text_detection/2025_05_07
- `s3://ocr_error_detection/2025_02_18`       -> MODEL_CACHE_DIR/ocr_error_detection/2025_02_18
- `hf://datalab-to/surya_layout2`（rf-detr 快速布局 + 阅读顺序头）
- `datalab-to/surya-ocr-2-gguf`（llama.cpp 后端的 VLM 权重 + mmproj）
- `datalab-to/surya-ocr-2`（vLLM 后端的 HF 权重，仅 `--with-vllm-weights` 时下载）

用法：
    python download_surya_models.py --dest <软件目录> --dir surya_models
    python download_surya_models.py --only s3          # 只补 s3 那两个
    python download_surya_models.py --only hf          # 只补 HF 那几个

进度以逐行 JSON 输出（`--progress_json`），与 Guaxi 转换脚本的协议一致，便于 Rust 侧
原样转发到设置页。
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "guaxi-surya-downloader"
S3_PREFIX = "s3://"
HF_PREFIX = "hf://"

# s3 上的模型目录（不含 s3:// 前缀），对应 surya.settings 里的 *MODEL_CHECKPOINT。
S3_CHECKPOINTS: List[Tuple[str, str]] = [
    ("DETECTOR_MODEL_CHECKPOINT", "text_detection/2025_05_07"),
    ("OCR_ERROR_MODEL_CHECKPOINT", "ocr_error_detection/2025_02_18"),
]

# HF 仓库。**不做 allow_patterns 过滤**：必须拉成"完整快照"。
#
# 原因：marker 运行时会进入 HF 离线模式（模型已在本地，避免每次转换都先去 HF API
# 核对 revision——大陆网络下那个请求经常抖动，会直接把整次转换打断）。而
# huggingface_hub 在离线模式下会拿下载时留下的仓库清单 `trees/<rev>.json` 逐个核对
# 本地文件，只要缺一个（哪怕是不参与推理的 LICENSE / README.md）就抛
# IncompleteSnapshotError。因此这里必须整仓拉取，多出来的几十 KB 换的是"离线可用"。
HF_CHECKPOINTS: List[Tuple[str, str, List[str]]] = [
    ("SURYA_GGUF_REPO", "datalab-to/surya-ocr-2-gguf", []),
    ("FAST_LAYOUT_MODEL_CHECKPOINT", "datalab-to/surya_layout2", []),
    ("FAST_ORDER_MODEL_CHECKPOINT", "datalab-to/surya_layout2", []),
]

# vLLM 后端用的 HF 权重（Windows 上我们用 llama.cpp，默认不下载）。
HF_VLLM_WEIGHTS = ("SURYA_MODEL_CHECKPOINT", "datalab-to/surya-ocr-2")

RETRIES = 3


def _emit(enabled: bool, **payload) -> None:
    if not enabled:
        return
    payload.setdefault("type", "model_download_progress")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _local_name(path: Path) -> str:
    return path.name


def _download_file(url: str, dest_path: Path, on_retry=None) -> None:
    """下载单个文件到 dest_path，带 `.part` 断点续传与重试。

    服务端支持 Range 时从已下载的字节数继续；不支持（返回 200）则从头重写。
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_error: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as resp:
                # 206 = 服务端接受续传；200 = 忽略 Range，需要从头写。
                resuming = resp.status == 206 and resume_from > 0
                mode = "ab" if resuming else "wb"
                if not resuming:
                    resume_from = 0
                with open(tmp_path, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            os.replace(tmp_path, dest_path)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt >= RETRIES:
                break
            delay = min(30, 2 ** attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay)

    raise RuntimeError(f"下载失败 {url}: {last_error}")


def _download_s3_directory(base_url: str, remote_dir: str, local_dir: Path, workers: int, progress: bool) -> None:
    """按 manifest.json 下载一个 s3 目录（与 surya.common.s3 的落地布局一致）。"""
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_url = _join_url(_join_url(base_url, remote_dir), "manifest.json")
    manifest_bytes = _http_get_bytes(manifest_url)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    files = manifest.get("files") or []

    # manifest.json 本身也必须落进目录：surya 的 check_manifest() 正是靠它判断“模型已完整”。
    # 少了它，surya 会认定模型不存在而重新下载——在 Windows 上还会撞上 shutil.move 的
    # “目标已存在”硬失败。写在这里（而不是下载循环里）是因为它不属于 manifest["files"]。
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.is_file():
        manifest_path.write_bytes(manifest_bytes)

    missing = [rel for rel in files if not (local_dir / rel).exists()]
    if not missing:
        _emit(progress, phase="s3", current=len(files), total=len(files),
              message=f"{remote_dir} 已完整，跳过")
        return

    _emit(progress, phase="s3", current=0, total=len(missing),
          message=f"下载 {remote_dir}（缺 {len(missing)}/{len(files)} 个文件）")

    done = 0
    counter_lock = threading.Lock()

    def _task(rel: str) -> None:
        nonlocal done
        url = _join_url(_join_url(base_url, remote_dir), rel)
        _download_file(url, local_dir / rel)
        with counter_lock:
            done += 1
            _emit(progress, phase="s3", current=done, total=len(missing),
                  message=f"{remote_dir}: {_local_name(Path(rel))}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_task, rel) for rel in missing]
        for fut in futures:
            fut.result()


def _download_hf_repo(repo_id: str, allow_patterns: List[str], progress: bool, label: str) -> None:
    """用 huggingface_hub 拉取 HF 仓库。

    走 `HF_ENDPOINT`（默认官方，可设 https://hf-mirror.com），并沿用 `HF_HOME`，
    因此文件会落在 surya 运行时期望的位置（surya 自己也用 snapshot_download 取 hf://）。
    huggingface_hub 自带断点续传与重试，这里只负责转发它的进度到 JSON。
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"缺少 huggingface_hub，无法下载 {repo_id}: {exc}") from exc

    _emit(progress, phase="hf", current=0, total=0, message=f"下载 {label}（{repo_id}）")
    snapshot_download(
        repo_id=repo_id,
        # 空列表视为“不加过滤”，即整仓拉取（保证离线模式可用，见上面的说明）。
        allow_patterns=allow_patterns or None,
        max_workers=8,
    )
    _emit(progress, phase="hf", current=1, total=1, message=f"{label} 就绪")


def _resolve_dest(args) -> Path:
    return (Path(args.dest).expanduser().resolve() / args.dir).resolve()


def _materialize_symlinks(root: Path) -> int:
    """把 HF 缓存里指回 `blobs/` 的符号链接替换成真实文件副本。

    为什么要这一步：huggingface_hub 的缓存布局是 `blobs/<hash>` 加
    `snapshots/<rev>/<name>` 符号链接。把这种目录打成 zip 分发时，链接很容易丢失
    （被解压工具跳过、或拷成一个只含目标路径的文本文件），对方机器上就会出现
    「目录看着在、模型实际不可用」这种最难排查的故障。物化一次之后，包在任何
    解压/拷贝工具下都能用。

    返回被替换的链接数量。
    """
    replaced = 0
    for path in root.rglob("*"):
        try:
            if not path.is_symlink():
                continue
            target = Path(os.path.realpath(path))
            if not target.is_file():
                continue
            tmp = path.with_name(path.name + ".materialize-part")
            shutil.copyfile(target, tmp)
            os.remove(path)
            os.replace(tmp, path)
            replaced += 1
        except Exception as exc:  # noqa: BLE001
            print(f"materialize failed for {path}: {exc}", file=sys.stderr)
    return replaced


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Surya models into a local folder, without importing surya."
    )
    parser.add_argument("--dest", default=".", help="Destination base directory (default: .).")
    parser.add_argument("--dir", default="surya_models", help="Subdir under --dest (default: surya_models).")
    parser.add_argument("--base-url", default="https://models.datalab.to", help="s3 模型基址。")
    parser.add_argument("--workers", type=int, default=10, help="并行下载线程数（s3 部分）。")
    parser.add_argument("--only", choices=["all", "s3", "hf"], default="all", help="只下载某一类。")
    parser.add_argument("--with-vllm-weights", action="store_true",
                        help="额外下载 vLLM 用的 HF 权重（datalab-to/surya-ocr-2）。")
    parser.add_argument("--materialize", action="store_true",
                        help="下载后把 HF 缓存里的符号链接替换成真实文件——"
                             "用于打包离线模型（链接在 zip 分发里极易丢失）。")
    parser.add_argument("--progress_json", action="store_true", help="逐行输出 JSON 进度。")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    progress = bool(args.progress_json)
    dest_dir = _resolve_dest(args)
    dest_dir.mkdir(parents=True, exist_ok=True)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    _emit(progress, phase="start", current=0, total=0,
          message=f"模型目录: {dest_dir}")
    print(f"Python       = {sys.executable}", flush=True)
    print(f"Base URL     = {args.base_url}", flush=True)
    print(f"HF endpoint  = {hf_endpoint or 'https://huggingface.co (default)'}", flush=True)
    print(f"Dest Dir     = {dest_dir}", flush=True)
    print("", flush=True)

    if args.only in ("all", "s3"):
        for name, remote in S3_CHECKPOINTS:
            print(f"[{name}] s3://{remote}", flush=True)
            _download_s3_directory(args.base_url, remote, dest_dir / remote, args.workers, progress)

    if args.only in ("all", "hf"):
        repos: List[Tuple[str, str, List[str]]] = list(HF_CHECKPOINTS)
        if args.with_vllm_weights:
            repos.append((HF_VLLM_WEIGHTS[0], HF_VLLM_WEIGHTS[1], ["*.json", "*.safetensors"]))
        # 同一个仓库可能出现多次（layout2 的主模型与 order 头），合并 pattern 只拉一遍。
        merged: dict = {}
        for name, repo, patterns in repos:
            merged.setdefault(repo, {"patterns": set(), "names": []})
            merged[repo]["patterns"].update(patterns)
            merged[repo]["names"].append(name)
        for repo, info in merged.items():
            print(f"[{', '.join(info['names'])}] hf://{repo}", flush=True)
            _download_hf_repo(repo, sorted(info["patterns"]), progress, repo)

    if args.materialize:
        replaced = _materialize_symlinks(dest_dir)
        print(f"Materialized {replaced} symlink(s) into real files.", flush=True)
        _emit(progress, phase="materialize", current=1, total=1,
              message=f"已物化 {replaced} 个符号链接")

    print("", flush=True)
    print(f"Done. Models are under: {dest_dir}", flush=True)
    _emit(progress, phase="done", current=1, total=1, message="模型下载完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
