# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  scripts/setup.ps1 — Dựng môi trường dev từ đầu, một lệnh duy nhất       ║
# ║                                                                          ║
# ║      .\scripts\setup.ps1              # cài từ lockfile (khuyến nghị)     ║
# ║      .\scripts\setup.ps1 -Loose       # cài từ requirements.txt          ║
# ║      .\scripts\setup.ps1 -SkipTorch   # bỏ qua bước cài torch            ║
# ║                                                                          ║
# ║  Script tạo .venv NGAY TRONG thư mục project (không dùng conda), nên      ║
# ║  không bao giờ còn cảnh "quên activate env nào".                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

param(
    [switch]$Loose,      # cài từ requirements.txt thay vì requirements.lock.txt
    [switch]$SkipTorch   # bỏ qua bước cài torch
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "=== RAG-pipeline-visualizer — dung moi truong dev ===" -ForegroundColor Cyan

# ── 1. uv ────────────────────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[1/4] uv chua co - dang cai..." -ForegroundColor Yellow
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
} else {
    Write-Host "[1/4] uv: $(uv --version)" -ForegroundColor Green
}

# ── 2. venv trên đúng phiên bản Python ───────────────────────────────────────
# uv đọc .python-version (3.11) và TỰ TẢI Python 3.11 nếu máy chưa có.
Write-Host "[2/4] Tao .venv tren Python 3.11..." -ForegroundColor Yellow
uv venv --python 3.11
if ($LASTEXITCODE -ne 0) { throw "uv venv that bai" }

# ── 3. Thư viện ──────────────────────────────────────────────────────────────
$reqFile = if ($Loose) { "requirements.txt" } else { "requirements.lock.txt" }
if (-not (Test-Path $reqFile)) {
    throw "Khong tim thay $reqFile. Neu thieu lockfile, chay lai voi -Loose."
}
Write-Host "[3/4] Cai thu vien tu $reqFile..." -ForegroundColor Yellow
uv pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "Cai thu vien that bai" }

# ── 4. torch (bước riêng, phụ thuộc phần cứng) ───────────────────────────────
if ($SkipTorch) {
    Write-Host "[4/4] Bo qua torch (-SkipTorch)." -ForegroundColor DarkGray
} else {
    Write-Host "[4/4] Cai torch dung voi GPU/driver cua may..." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe scripts\install_torch.py --apply
}

Write-Host ""
Write-Host "=== XONG ===" -ForegroundColor Green
Write-Host "Kich hoat moi truong:  .\.venv\Scripts\Activate.ps1"
Write-Host "Chay app:              streamlit run app.py"
Write-Host "Kiem tra moi truong:   python scripts\smoke_test.py --pdf data\test.pdf --mode single --offline"
