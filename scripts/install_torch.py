#!/usr/bin/env python3
"""
scripts/install_torch.py
========================
Tự dò phần cứng và cài bản PyTorch phù hợp nhất.

VÌ SAO CẦN SCRIPT NÀY
---------------------
`pip install torch` cho kết quả KHÁC NHAU tuỳ hệ điều hành:

  Linux   → PyPI trả bản CUDA (dùng được GPU)
  Windows → PyPI trả bản CPU-only ⚠️ GPU nằm im, không báo lỗi gì
  macOS   → bản CPU/MPS (đúng, vì macOS không có CUDA)

Ngoài ra bản CUDA trên PyPI được chốt ở MỘT phiên bản CUDA cố định. Nếu driver
của bạn cũ hơn, torch cài xong vẫn import được nhưng `torch.cuda.is_available()`
trả False — im lặng chạy CPU.

NGUYÊN LÝ DÒ
------------
`nvidia-smi` báo "CUDA Version: X.Y" = mức CUDA cao nhất mà DRIVER hỗ trợ,
không phải toolkit đã cài. Đây mới là con số cần đọc, vì PyTorch đóng gói sẵn
CUDA runtime bên trong — bạn KHÔNG cần cài CUDA toolkit riêng, chỉ cần driver
đủ mới. Script chọn bản PyTorch có CUDA <= con số đó.

CÁCH DÙNG
---------
    python scripts/install_torch.py              # chỉ dò và in lệnh, không cài
    python scripts/install_torch.py --apply      # dò rồi cài luôn
    python scripts/install_torch.py --check      # kiểm tra torch hiện tại
    python scripts/install_torch.py --apply --force-cpu    # ép bản CPU
    python scripts/install_torch.py --apply --cuda cu124   # ép CUDA cụ thể
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

# Console Windows mặc định là cp1252 → mọi lệnh print tiếng Việt/emoji dưới đây
# sẽ ném UnicodeEncodeError và script chết ngay dòng in đầu tiên.
# Ép stdout/stderr về UTF-8 TRƯỚC khi in bất cứ thứ gì.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TORCH_INDEX = "https://download.pytorch.org/whl"

# Các kênh CUDA xếp từ MỚI đến CŨ. Script sẽ probe index thật để biết kênh nào
# còn sống, nên khi PyTorch ra cu13x bạn chỉ cần thêm một dòng vào đầu list.
CUDA_CHANNELS: list[tuple[str, tuple[int, int]]] = [
    ("cu128", (12, 8)),
    ("cu126", (12, 6)),
    ("cu124", (12, 4)),
    ("cu121", (12, 1)),
    ("cu118", (11, 8)),
]

ROCM_CHANNELS = ["rocm6.2", "rocm6.1"]

C_RESET, C_RED, C_GREEN, C_YELLOW, C_BLUE, C_DIM = (
    "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[2m"
)
if not sys.stdout.isatty():
    C_RESET = C_RED = C_GREEN = C_YELLOW = C_BLUE = C_DIM = ""


# ═════════════════════════════════════════════════════════════════════════════
# Dò phần cứng
# ═════════════════════════════════════════════════════════════════════════════

def find_nvidia_smi() -> str | None:
    """Tìm nvidia-smi. Trên Windows nó thường không nằm trong PATH."""
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if platform.system() == "Windows":
        for cand in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            import os
            if os.path.exists(cand):
                return cand
    return None


def probe_nvidia() -> dict | None:
    """
    Trả về {'gpus': [...], 'driver': '5xx.xx', 'cuda': (12, 4)} hoặc None.
    """
    exe = find_nvidia_smi()
    if not exe:
        return None

    # Tên GPU + driver: lấy bằng --query-gpu cho ổn định (không phụ thuộc layout)
    try:
        r = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None

    gpus, driver = [], None
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory": parts[2]})
            driver = driver or parts[1]

    # CUDA version: chỉ có ở output mặc định, không có trong --query-gpu
    cuda = None
    try:
        r2 = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r2.stdout)
        if m:
            cuda = (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass

    if not gpus:
        return None
    return {"gpus": gpus, "driver": driver, "cuda": cuda}


def probe_rocm() -> dict | None:
    """Dò GPU AMD (chỉ Linux)."""
    if platform.system() != "Linux":
        return None
    exe = shutil.which("rocminfo") or shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        names = re.findall(r"Marketing Name:\s*(.+)", r.stdout)
        return {"gpus": [{"name": n.strip(), "memory": "?"} for n in names[:4]] or
                        [{"name": "AMD GPU", "memory": "?"}]}
    except Exception:
        return None


def channel_exists(channel: str, timeout: float = 8.0) -> bool:
    """
    Probe index thật thay vì tin vào bảng hard-code. Nhờ vậy script không lỗi
    thời khi PyTorch thêm/bỏ kênh CUDA.
    """
    url = f"{TORCH_INDEX}/{channel}/"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception:
        # Không có mạng / bị chặn → không kết luận được, cứ coi là có
        return True


# ═════════════════════════════════════════════════════════════════════════════
# Quyết định
# ═════════════════════════════════════════════════════════════════════════════

def decide(force_cpu: bool, force_cuda: str | None, no_probe: bool) -> dict:
    system = platform.system()
    py = f"{sys.version_info.major}.{sys.version_info.minor}"

    if force_cpu:
        return {"channel": "cpu", "why": "bạn yêu cầu --force-cpu",
                "device": "CPU", "system": system, "python": py}

    if force_cuda:
        return {"channel": force_cuda, "why": f"bạn yêu cầu --cuda {force_cuda}",
                "device": "NVIDIA GPU", "system": system, "python": py}

    if system == "Darwin":
        chip = platform.machine()
        mps = chip == "arm64"
        return {
            "channel": None,          # macOS: dùng wheel mặc định của PyPI
            "device": "Apple MPS" if mps else "CPU",
            "why": ("macOS không có CUDA. Apple Silicon dùng backend MPS, "
                    "có sẵn trong wheel mặc định."
                    if mps else "Mac Intel — chỉ có CPU."),
            "system": system, "python": py,
        }

    nv = probe_nvidia()
    if nv:
        cuda = nv["cuda"]
        info = {
            "device": nv["gpus"][0]["name"],
            "gpus": nv["gpus"],
            "driver": nv["driver"],
            "driver_cuda": cuda,
            "system": system, "python": py,
        }
        if not cuda:
            info.update(channel="cu121",
                        why="Thấy GPU NVIDIA nhưng không đọc được CUDA version "
                            "từ nvidia-smi → chọn cu121 cho an toàn.")
            return info

        for chan, ver in CUDA_CHANNELS:
            if ver <= cuda:
                if no_probe or channel_exists(chan):
                    info.update(
                        channel=chan,
                        why=f"Driver {nv['driver']} hỗ trợ tới CUDA "
                            f"{cuda[0]}.{cuda[1]} → {chan} là bản mới nhất dùng được.")
                    return info

        info.update(channel="cpu",
                    why=f"Driver chỉ hỗ trợ CUDA {cuda[0]}.{cuda[1]}, "
                        f"cũ hơn mọi bản PyTorch hiện có. "
                        f"NÊN NÂNG DRIVER thay vì dùng CPU.")
        return info

    amd = probe_rocm()
    if amd:
        chan = next((c for c in ROCM_CHANNELS if no_probe or channel_exists(c)),
                    ROCM_CHANNELS[0])
        return {"channel": chan, "device": amd["gpus"][0]["name"],
                "gpus": amd["gpus"],
                "why": f"Thấy GPU AMD → dùng kênh ROCm ({chan}). "
                       f"ROCm chỉ hỗ trợ Linux.",
                "system": system, "python": py}

    return {"channel": "cpu", "device": "CPU",
            "why": "Không tìm thấy nvidia-smi hay rocminfo → không có GPU dùng được.",
            "system": system, "python": py}


def _has_pip() -> bool:
    """Interpreter hiện tại có module pip không?

    venv tạo bằng ``uv venv`` KHÔNG có pip (uv dùng resolver riêng). Khi đó
    ``python -m pip install`` sẽ chết với "No module named pip".
    """
    import importlib.util
    return importlib.util.find_spec("pip") is not None


def _uv_path() -> str | None:
    """Đường dẫn tới uv nếu có trên PATH."""
    return shutil.which("uv")


def _installer_prefix(action: str) -> list[str] | None:
    """
    Chọn trình cài đặt phù hợp với môi trường hiện tại.

    Ưu tiên pip (có sẵn ở venv/conda truyền thống). Nếu không có pip thì dùng
    ``uv pip --python <interpreter>`` — uv cài vào ĐÚNG interpreter đang chạy
    script, không phải vào môi trường mặc định của uv.

    Tham số
    -------
    action : "install" | "uninstall"

    Trả về
    ------
    Phần đầu của lệnh, hoặc None nếu không có trình cài đặt nào.
    """
    if _has_pip():
        return [sys.executable, "-m", "pip", action]
    uv = _uv_path()
    if uv:
        return [uv, "pip", action, "--python", sys.executable]
    return None


def build_command(channel: str | None) -> list[str] | None:
    prefix = _installer_prefix("install")
    if prefix is None:
        return None
    base = prefix + ["torch", "torchvision"]
    if channel is None:
        return base
    return base + ["--index-url", f"{TORCH_INDEX}/{channel}"]


# ═════════════════════════════════════════════════════════════════════════════
# Kiểm tra sau khi cài
# ═════════════════════════════════════════════════════════════════════════════

def check_torch() -> int:
    # Cảnh báo nếu torch đã nằm trong sys.modules TRƯỚC khi hàm này chạy —
    # khi đó số liệu bên dưới có thể là của bản cũ còn trong RAM.
    if "torch" in sys.modules:
        print(f"{C_YELLOW}⚠️  torch đã được nạp sẵn trong tiến trình này — "
              f"số liệu dưới đây có thể là bản CŨ.{C_RESET}")
        print(f"{C_DIM}   Chạy lại bằng lệnh riêng để có kết quả chính xác:"
              f"\n     python {sys.argv[0]} --check{C_RESET}\n")
    try:
        import torch
    except ImportError:
        print(f"{C_YELLOW}torch chưa được cài.{C_RESET}")
        return 1
    except Exception as exc:
        # Bản cài dở (tải 2.6 GB bị đứt) import được nhưng lỗi khi khởi tạo
        print(f"{C_RED}torch đã cài nhưng KHÔNG import được.{C_RESET}")
        print(f"  Nguyên nhân: {type(exc).__name__}: {exc}")
        print(f"  Cài lại: pip uninstall -y torch torchvision && "
              f"python {sys.argv[0]} --apply")
        return 1

    version = getattr(torch, "__version__", None)
    if version is None:
        print(f"{C_RED}torch import được nhưng thiếu __version__ — "
              f"bản cài hỏng.{C_RESET}")
        print(f"  Cài lại: pip uninstall -y torch torchvision && "
              f"python {sys.argv[0]} --apply")
        return 1

    print(f"  torch            : {version}")
    built = getattr(torch.version, "cuda", None)
    print(f"  CUDA build       : {built or '(bản CPU-only)'}")
    hip = getattr(torch.version, "hip", None)
    if hip:
        print(f"  ROCm build       : {hip}")

    avail = torch.cuda.is_available()
    color = C_GREEN if avail else C_YELLOW
    print(f"  cuda.is_available: {color}{avail}{C_RESET}")

    if avail:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name} · {p.total_memory / 1024**3:.1f} GB "
                  f"· compute {p.major}.{p.minor}")
        return 0

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  {C_GREEN}Apple MPS khả dụng.{C_RESET}")
        return 0

    print()
    if built is None:
        print(f"{C_YELLOW}Đang dùng torch bản CPU-only.{C_RESET}")
        print("  Nếu máy CÓ GPU NVIDIA, chạy:")
        print(f"    python {sys.argv[0]} --apply")
    else:
        print(f"{C_YELLOW}torch có CUDA {built} nhưng không thấy GPU.{C_RESET}")
        print("  Nguyên nhân thường gặp:")
        print("    · Driver NVIDIA cũ hơn CUDA của bản torch → nâng driver")
        print("    · Chạy trong container thiếu --gpus all")
        print("    · WSL2 chưa cài driver NVIDIA cho Windows host")
    return 1


# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dò GPU/CUDA và cài PyTorch phù hợp")
    ap.add_argument("--apply", action="store_true", help="Cài thật (mặc định chỉ in lệnh)")
    ap.add_argument("--check", action="store_true", help="Kiểm tra torch đang cài")
    ap.add_argument("--force-cpu", action="store_true", help="Ép cài bản CPU")
    ap.add_argument("--cuda", metavar="cuXXX", help="Ép kênh CUDA, vd: cu124")
    ap.add_argument("--no-probe", action="store_true",
                    help="Không probe index PyTorch (dùng khi mạng chặn)")
    args = ap.parse_args()

    if args.check:
        print(f"{C_BLUE}Trạng thái PyTorch{C_RESET}")
        return check_torch()

    print(f"{C_BLUE}Dò phần cứng{C_RESET}")
    d = decide(args.force_cpu, args.cuda, args.no_probe)

    print(f"  Hệ điều hành     : {d['system']}")
    print(f"  Python           : {d['python']}")
    print(f"  Thiết bị         : {C_GREEN}{d['device']}{C_RESET}")
    if d.get("driver"):
        print(f"  Driver NVIDIA    : {d['driver']}")
    if d.get("driver_cuda"):
        c = d["driver_cuda"]
        print(f"  Driver hỗ trợ CUDA tới: {c[0]}.{c[1]}")
    for g in d.get("gpus", [])[1:]:
        print(f"                     + {g['name']} ({g['memory']})")

    print(f"\n  {C_DIM}{d['why']}{C_RESET}")

    if d["system"] not in ("Linux", "Darwin", "Windows"):
        print(f"{C_YELLOW}  Hệ điều hành lạ — hãy tự chọn bản torch.{C_RESET}")

    cmd = build_command(d["channel"])
    if cmd is None:
        print(f"\n{C_RED}Không tìm thấy trình cài đặt.{C_RESET}")
        print("  Môi trường này không có `pip`, và cũng không thấy `uv` trên PATH.")
        print("  venv tạo bằng `uv venv` không kèm pip. Chọn một trong hai cách:")
        print("    · Cài uv:            https://astral.sh/uv")
        print("    · Hoặc thêm pip vào venv:  uv venv --seed --python 3.11")
        return 1

    print(f"\n{C_BLUE}Lệnh cài{C_RESET}")
    print(f"  {' '.join(cmd)}")

    if not args.apply:
        print(f"\n{C_DIM}Chạy lại với --apply để cài thật.{C_RESET}")
        return 0

    # torch cài sẵn ở biến thể khác sẽ không tự bị thay → gỡ trước.
    #
    # Dùng find_spec chứ KHÔNG `import torch`: import sẽ nạp module vào
    # sys.modules, và mọi lần `import torch` sau đó trong cùng tiến trình sẽ
    # trả về object CŨ trong RAM, kể cả khi file trên đĩa đã bị pip thay.
    # Đó chính là lỗi khiến phần "Kiểm tra kết quả" báo phiên bản trước khi cài.
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        print(f"\n{C_YELLOW}Đã có torch — gỡ trước để tránh trộn biến thể.{C_RESET}")
        uninstall = _installer_prefix("uninstall")
        if uninstall:
            # uv pip uninstall không có/không cần cờ -y; pip thì cần.
            flags = [] if uninstall[0] != sys.executable else ["-y"]
            subprocess.run(uninstall + flags +
                           ["torch", "torchvision", "torchaudio"], check=False)

    print(f"\n{C_BLUE}Đang cài...{C_RESET} (bản CUDA nặng ~2.5 GB, kiên nhẫn)\n")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n{C_RED}Cài thất bại.{C_RESET}")
        print("  Thử: --force-cpu để dùng bản CPU, hoặc --cuda cu121 cho driver cũ.")
        return r.returncode

    # Kiểm tra trong TIẾN TRÌNH MỚI. Bắt buộc: torch chứa extension module
    # (.so/.pyd) đã nạp vào tiến trình hiện tại; file trên đĩa vừa bị thay
    # nhưng bản trong RAM thì không. importlib.reload() cũng không cứu được.
    # Chỉ interpreter mới đọc đúng những gì vừa cài.
    print(f"\n{C_BLUE}Kiểm tra kết quả{C_RESET} {C_DIM}(chạy trong tiến trình mới){C_RESET}")
    import os
    _self = os.path.abspath(__file__)
    r2 = subprocess.run([sys.executable, _self, "--check"])
    return r2.returncode


if __name__ == "__main__":
    sys.exit(main())
