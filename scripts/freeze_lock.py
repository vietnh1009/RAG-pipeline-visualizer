#!/usr/bin/env python
"""
scripts/freeze_lock.py
======================
Sinh lại ``requirements.lock.txt`` từ MỘT môi trường đang chạy tốt.

Vì sao cần file này
-------------------
``requirements.txt`` chỉ pin dependency TRỰC TIẾP. Dependency gián tiếp
(transitive) vẫn được resolve lại mỗi lần cài → hai máy cùng lệnh cài vẫn có
thể ra hai bộ thư viện khác nhau. Đó là nguyên nhân chính của tình trạng
"máy này chạy được, máy kia lỗi".

``requirements.lock.txt`` chụp lại TOÀN BỘ cây dependency của môi trường đang
chạy tốt, nên cài lại ở đâu cũng ra đúng bộ đó.

Cách dùng
---------
    python scripts/freeze_lock.py                     # dùng Python đang chạy
    python scripts/freeze_lock.py --python <path>     # chỉ định interpreter khác
    python scripts/freeze_lock.py --output custom.txt

Ví dụ (Windows, môi trường conda):
    python scripts/freeze_lock.py --python F:/miniconda3/envs/rag_visualizer/python.exe

Ghi chú
-------
torch / torchvision / torchaudio / nvidia-* / triton bị LOẠI khỏi lockfile.
Chúng phụ thuộc phần cứng và index riêng của PyTorch; pin cứng ở đây sẽ hỏng
trên máy khác. Luôn cài chúng ở bước cuối bằng:

    python scripts/install_torch.py --apply
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

# Console Windows mặc định là cp1252 → in tiếng Việt/emoji sẽ ném
# UnicodeEncodeError. Ép stdout/stderr về UTF-8 trước khi in bất cứ thứ gì.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):   # stream đã bị thay thế / không hỗ trợ
        pass

# Các package phụ thuộc phần cứng — không đưa vào lockfile.
EXCLUDE = re.compile(r"^(torch|torchvision|torchaudio|nvidia|triton)", re.IGNORECASE)

HEADER = """\
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  requirements.lock.txt — KHOÁ TOÀN BỘ dependency (kể cả transitive)      ║
# ║                                                                          ║
# ║  SINH TỰ ĐỘNG — đừng sửa tay.                                            ║
# ║                                                                          ║
# ║  Cách dùng:                                                              ║
# ║      uv pip install -r requirements.lock.txt      (nhanh, khuyến nghị)   ║
# ║      pip install -r requirements.lock.txt         (chậm hơn)             ║
# ║                                                                          ║
# ║  KHÔNG chứa torch — cài riêng ở bước cuối:                               ║
# ║      python scripts/install_torch.py --apply                             ║
# ║                                                                          ║
# ║  Sinh lại:  python scripts/freeze_lock.py                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Nguồn      : {source}
# Python     : {pyver}
# Sinh lúc   : {date}
"""


def _python_version(exe: str) -> str:
    """Lấy chuỗi phiên bản Python của interpreter chỉ định."""
    out = subprocess.run(
        [exe, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _pip_check(exe: str) -> list[str]:
    """
    Chạy ``pip check`` trên môi trường nguồn.

    Môi trường "chạy được" vẫn có thể chứa mâu thuẫn metadata mà pip bỏ qua
    lúc cài (ví dụ: unstructured-client yêu cầu pypdf>=6.2.0 trong khi env có
    pypdf 5.1.0). pip vẫn chạy, nhưng resolver nghiêm ngặt như uv sẽ TỪ CHỐI
    cài lại lockfile đó. Phải phát hiện ngay tại bước sinh lock.

    Trả về
    ------
    Danh sách dòng mô tả xung đột (rỗng nếu sạch).
    """
    out = subprocess.run([exe, "-m", "pip", "check"], capture_output=True, text=True)
    if out.returncode == 0:
        return []
    return [l for l in out.stdout.splitlines() if l.strip()]


def _freeze(exe: str) -> list[str]:
    """Chạy ``pip freeze`` và lọc bỏ package phụ thuộc phần cứng."""
    out = subprocess.run(
        [exe, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    )
    lines = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if EXCLUDE.match(line):
            continue
        if "@ file://" in line:          # cài từ đường dẫn local — không tái lập được
            continue
        lines.append(line)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Sinh requirements.lock.txt từ môi trường đang chạy tốt.")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter nguồn (mặc định: Python đang chạy script này)")
    ap.add_argument("--output", default="requirements.lock.txt",
                    help="File đích (mặc định: requirements.lock.txt)")
    ap.add_argument("--allow-conflicts", action="store_true",
                    help="Vẫn sinh lock dù `pip check` báo xung đột (không khuyến nghị)")
    args = ap.parse_args()

    exe = args.python
    pyver = _python_version(exe)

    if not pyver.startswith(("3.10", "3.11", "3.12")):
        print(f"⚠️  Interpreter nguồn là Python {pyver} — project chỉ hỗ trợ 3.10–3.12.")
        print("    Lockfile sinh ra từ đây nhiều khả năng KHÔNG cài được. Dừng lại.")
        return 1

    conflicts = _pip_check(exe)
    if conflicts:
        print("⚠️  Môi trường nguồn có xung đột dependency:")
        for c in conflicts:
            print(f"     · {c}")
        print()
        print("    Lockfile sinh từ đây sẽ KHÔNG cài lại được bằng resolver nghiêm ngặt")
        print("    (uv, pip --strict). Sửa xung đột rồi chạy lại, hoặc dùng --allow-conflicts")
        print("    rồi tự sửa dòng tương ứng trong lockfile.")
        if not args.allow_conflicts:
            return 1
        print("    → --allow-conflicts được bật, vẫn tiếp tục.\n")

    packages = _freeze(exe)
    if not packages:
        print("❌ pip freeze không trả về gói nào. Sai interpreter?")
        return 1

    header = HEADER.format(
        source=exe,
        pyver=pyver,
        date=datetime.date.today().isoformat(),
    )
    Path(args.output).write_text(header + "\n" + "\n".join(packages) + "\n", encoding="utf-8")

    print(f"✅ Đã ghi {args.output} — {len(packages)} gói (Python {pyver}).")
    print("   torch/torchvision bị loại khỏi lock. Sau khi cài lock, chạy:")
    print("       python scripts/install_torch.py --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
