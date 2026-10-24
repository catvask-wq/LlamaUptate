import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


app = FastAPI()


# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PATH_FILE = BASE_DIR / "llama_path.txt"

API_RELEASES = (
    "https://api.github.com/repos/ggml-org/llama.cpp/releases"
    "?per_page=100"
)

MANAGER_HOST = "127.0.0.1"
MANAGER_PORT = 8000

UA = {
    "User-Agent": "llama-manager-updater/7.0",
    "Accept": "application/vnd.github+json",
}


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def read_text(path: Path):
    try:
        value = path.read_text(
            encoding="utf-8-sig"
        ).strip()

        return value or None

    except (OSError, UnicodeDecodeError):
        return None


def get_saved_folder():
    """
    Читает llama_path.txt.

    В файле может находиться:
    - путь к папке;
    - путь непосредственно к llama-server.exe.
    """

    value = read_text(PATH_FILE)

    if not value:
        return None

    value = value.strip().strip('"').strip("'")

    path = Path(value)

    if path.is_file():
        return path.parent

    if path.is_dir():
        return path

    return None


def find_server_exe(folder: Path):
    if not folder:
        return None

    direct_candidates = (
        folder / "llama-server.exe",
        folder / "bin" / "llama-server.exe",
    )

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    try:
        for candidate in folder.rglob(
            "llama-server.exe"
        ):
            if candidate.is_file():
                return candidate

    except OSError:
        pass

    return None


def find_optional_exe(folder: Path, names):
    """
    Ищет дополнительные программы:
    llama-cli.exe и другие.
    """

    if not folder:
        return None

    for name in names:
        candidates = (
            folder / name,
            folder / "bin" / name,
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

    try:
        for name in names:
            for candidate in folder.rglob(name):
                if candidate.is_file():
                    return candidate

    except OSError:
        pass

    return None


def run_capture(command, cwd=None, timeout=30):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            creationflags=getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            ),
        )

        output = (
            f"{result.stdout or ''}\n"
            f"{result.stderr or ''}"
        ).strip()

        return result.returncode, output

    except Exception as exc:
        return -1, f"(ошибка запуска: {exc})"


# ============================================================
# ПАРСИНГ ВЕРСИИ LLAMA-SERVER
# ============================================================

VERSION_RE = re.compile(
    r"""
    \bversion\s*:\s*
    (?P<version>[^\r\n(]+?)
    \s*\(\s*
    build\s*[:=]?\s*
    (?P<build>\d+)
    \s*,\s*
    commit\s*[:=]?\s*
    [`'" ]*
    (?P<commit>[0-9a-fA-F]{6,40})
    """,
    re.IGNORECASE | re.VERBOSE,
)


OLD_VERSION_RE = re.compile(
    r"""
    \bversion\s*:\s*
    (?:b)?
    (?P<build>\d+)
    \s*\(\s*
    (?:commit\s*)?
    [`'" ]*
    (?P<commit>[0-9a-fA-F]{6,40})
    """,
    re.IGNORECASE | re.VERBOSE,
)


BUILD_RE = re.compile(
    r"""
    \bbuild\s*[:=]?\s*
    (?P<build>\d+)
    \s*(?:,|\()\s*
    (?:commit\s*[:=]?\s*)?
    [`'" ]*
    (?P<commit>[0-9a-fA-F]{6,40})
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_version_output(output: str):
    """
    Поддерживает формат:

        version: 0.7.0-dev
        (build 11849, commit 0ae0e1976)
    """

    if not output:
        return None

    match = VERSION_RE.search(output)

    if match:
        groups = match.groupdict()

        try:
            build_number = int(groups["build"])
        except (TypeError, ValueError):
            return None

        version_name = (
            groups.get("version") or ""
        ).strip().strip("`'\" ")

        commit = (
            groups.get("commit") or ""
        ).strip()

        return {
            "build": build_number,
            "tag": f"b{build_number}",
            "commit": commit,
            "version": version_name or None,
            "display": (
                f"{version_name} "
                f"(build {build_number}, commit {commit})"
            ),
        }

    match = OLD_VERSION_RE.search(output)

    if match:
        groups = match.groupdict()

        try:
            build_number = int(groups["build"])
        except (TypeError, ValueError):
            return None

        commit = (
            groups.get("commit") or ""
        ).strip()

        return {
            "build": build_number,
            "tag": f"b{build_number}",
            "commit": commit,
            "version": None,
            "display": (
                f"b{build_number} "
                f"(commit {commit})"
            ),
        }

    match = BUILD_RE.search(output)

    if match:
        groups = match.groupdict()

        try:
            build_number = int(groups["build"])
        except (TypeError, ValueError):
            return None

        commit = (
            groups.get("commit") or ""
        ).strip()

        return {
            "build": build_number,
            "tag": f"b{build_number}",
            "commit": commit,
            "version": None,
            "display": (
                f"b{build_number} "
                f"(commit {commit})"
            ),
        }

    return None


def get_server_version(server: Path):
    """
    Получает версию непосредственно из llama-server.exe.
    """

    last_output = ""

    for args in (
        ["--version"],
        ["--help"],
    ):
        return_code, output = run_capture(
            [str(server)] + args,
            cwd=server.parent,
            timeout=20,
        )

        last_output = output

        parsed = parse_version_output(output)

        if parsed:
            return parsed, output

    return None, last_output[:1500]


# ============================================================
# GPU
# ============================================================

def get_nvidia_info():
    return_code, output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        timeout=15,
    )

    if return_code != 0:
        return None

    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    name, _, driver = lines[0].partition(",")

    name = name.strip()
    driver = driver.strip()

    major_text = driver.split(".", 1)[0]

    if major_text.isdigit():
        driver_major = int(major_text)
    else:
        driver_major = 0

    if driver_major >= 580:
        flavor = "cuda-cu13.4"
    else:
        flavor = "cuda-cu12.4"

    return {
        "vendor": "NVIDIA",
        "name": name,
        "driver": driver,
        "flavor": flavor,
        "description": (
            f"{name}, драйвер {driver}, "
            f"выбрана {flavor}"
        ),
    }


def get_gpu_names():
    commands = (
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "(Get-CimInstance "
                "Win32_VideoController).Name"
            ),
        ],
        [
            "wmic",
            "path",
            "win32_videocontroller",
            "get",
            "name",
        ],
    )

    for command in commands:
        return_code, output = run_capture(
            command,
            timeout=25,
        )

        if return_code != 0:
            continue

        names = []

        for line in output.splitlines():
            line = line.strip()

            if not line:
                continue

            if line.lower() == "name":
                continue

            names.append(line)

        if names:
            return names

    return []


def detect_gpu():
    """
    Правила:

    NVIDIA -> CUDA
    AMD -> Vulkan
    Intel -> Vulkan
    прочее -> CPU
    """

    nvidia = get_nvidia_info()

    if nvidia:
        return nvidia

    names = get_gpu_names()

    for name in names:
        if re.search(
            r"NVIDIA|GeForce|Quadro|RTX",
            name,
            re.IGNORECASE,
        ):
            return {
                "vendor": "NVIDIA",
                "name": name,
                "driver": None,
                "flavor": "cuda-cu12.4",
                "description": (
                    f"{name}, выбрана CUDA 12.4"
                ),
            }

    for name in names:
        if re.search(
            r"AMD|Radeon|RX|FirePro",
            name,
            re.IGNORECASE,
        ):
            return {
                "vendor": "AMD",
                "name": name,
                "driver": None,
                "flavor": "vulkan",
                "description": (
                    f"{name}, выбрана Vulkan"
                ),
            }

    for name in names:
        if re.search(
            r"Intel|Arc",
            name,
            re.IGNORECASE,
        ):
            return {
                "vendor": "Intel",
                "name": name,
                "driver": None,
                "flavor": "vulkan",
                "description": (
                    f"{name}, выбрана Vulkan"
                ),
            }

    return {
        "vendor": "unknown",
        "name": "GPU не определена",
        "driver": None,
        "flavor": "cpu",
        "description": (
            "GPU не определена, выбрана CPU-сборка"
        ),
    }


# ============================================================
# ОПРЕДЕЛЕНИЕ BACKEND
# ============================================================

BACKEND_PRIORITY = (
    "Vulkan",
    "CUDA",
    "HIP/ROCm",
    "SYCL",
)


def backends_from_text(text: str):
    """
    Ищет backend в --help / -h.
    """

    if not text:
        return []

    lower = text.lower()
    found = []

    patterns = (
        (
            "Vulkan",
            (
                r"\bggml[-_ ]?vulkan\b",
                r"\bvulkan\b",
                r"backend\s*[:=]\s*vulkan",
            ),
        ),
        (
            "CUDA",
            (
                r"\bggml[-_ ]?cuda\b",
                r"\bcuda\b",
                r"backend\s*[:=]\s*cuda",
            ),
        ),
        (
            "HIP/ROCm",
            (
                r"\bggml[-_ ]?hip\b",
                r"\bhip\b",
                r"\brocm\b",
                r"backend\s*[:=]\s*hip",
            ),
        ),
        (
            "SYCL",
            (
                r"\bggml[-_ ]?sycl\b",
                r"\bsycl\b",
                r"backend\s*[:=]\s*sycl",
            ),
        ),
    )

    for backend, backend_patterns in patterns:
        for pattern in backend_patterns:
            if re.search(
                pattern,
                lower,
                re.IGNORECASE,
            ):
                found.append(backend)
                break

    return found


def detect_backend_details(folder: Path):
    """
    Определяет backend несколькими способами:

    1. По ggml DLL.
    2. По llama-server.exe --help.
    3. По llama-cli.exe -h.
    4. По llama-cli.exe --help.
    """

    if not folder:
        return "Unknown", []

    detected = []
    sources = []

    def add_backend(backend, source):
        if backend not in detected:
            detected.append(backend)

        sources.append(
            f"{source}: {backend}"
        )

    # --------------------------------------------------------
    # Способ 1. DLL
    # --------------------------------------------------------

    try:
        files = [
            path
            for path in folder.rglob("*")
            if path.is_file()
        ]

    except OSError:
        files = []

    file_names = [
        path.name.lower()
        for path in files
    ]

    if any(
        "ggml-vulkan" in name
        or "llama-vulkan" in name
        for name in file_names
    ):
        add_backend(
            "Vulkan",
            "DLL",
        )

    if any(
        "ggml-cuda" in name
        or "llama-cuda" in name
        for name in file_names
    ):
        add_backend(
            "CUDA",
            "DLL",
        )

    if any(
        "ggml-hip" in name
        or "llama-hip" in name
        or "rocm" in name
        for name in file_names
    ):
        add_backend(
            "HIP/ROCm",
            "DLL",
        )

    if any(
        "ggml-sycl" in name
        or "llama-sycl" in name
        for name in file_names
    ):
        add_backend(
            "SYCL",
            "DLL",
        )

    # --------------------------------------------------------
    # Способ 2. llama-server.exe --help
    # --------------------------------------------------------

    server = find_server_exe(folder)

    if server:
        return_code, output = run_capture(
            [
                str(server),
                "--help",
            ],
            cwd=server.parent,
            timeout=20,
        )

        help_backends = backends_from_text(
            output
        )

        for backend in help_backends:
            add_backend(
                backend,
                "llama-server.exe --help",
            )

    # --------------------------------------------------------
    # Способ 3 и 4. llama-cli.exe
    # --------------------------------------------------------

    cli = find_optional_exe(
        folder,
        [
            "llama-cli.exe",
            "llama-cli",
        ],
    )

    if cli:
        for args in (
            ["-h"],
            ["--help"],
        ):
            return_code, output = run_capture(
                [str(cli)] + args,
                cwd=cli.parent,
                timeout=20,
            )

            cli_backends = backends_from_text(
                output
            )

            for backend in cli_backends:
                add_backend(
                    backend,
                    f"{cli.name} {' '.join(args)}",
                )

    if not detected:
        return "Unknown", sources

    # Если обнаружен Vulkan, выбираем его первым.
    # Это важно для AMD-сборок.
    for backend in BACKEND_PRIORITY:
        if backend in detected:
            return backend, sources

    return detected[0], sources


def detect_installed_backend(folder: Path):
    backend, sources = detect_backend_details(
        folder
    )

    return backend


def expected_backend(flavor: str):
    flavor = (flavor or "").lower()

    if flavor.startswith("cuda"):
        return "CUDA"

    if flavor == "vulkan":
        return "Vulkan"

    if flavor in (
        "hip",
        "hip-radeon",
    ):
        return "HIP/ROCm"

    if flavor == "sycl":
        return "SYCL"

    if flavor == "cpu":
        return "CPU"

    return "Unknown"


def backend_matches(flavor: str, installed_backend: str):
    expected = expected_backend(flavor)

    if expected == "CPU":
        return installed_backend in (
            "CPU",
            "Unknown",
        )

    return expected == installed_backend


# ============================================================
# GITHUB ASSET
# ============================================================

def get_build_from_tag(tag: str):
    """
    b11849 -> 11849
    """

    if not tag:
        return None

    match = re.search(
        r"(?<!\d)b(\d+)(?!\d)",
        tag,
        re.IGNORECASE,
    )

    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def cuda_version_from_name(name: str):
    match = re.search(
        r"-cuda-(?:cu)?"
        r"([0-9]+(?:\.[0-9]+)?)",
        name,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def pick_cudart_asset(names, cuda_version):
    if not cuda_version:
        return None

    escaped_version = re.escape(
        cuda_version
    )

    for name in names:
        lower_name = name.lower()

        if not lower_name.startswith(
            "cudart-"
        ):
            continue

        if not lower_name.endswith(
            "-x64.zip"
        ):
            continue

        if re.search(
            rf"(?:cuda-)?(?:cu)?"
            rf"{escaped_version}"
            rf"(?:[-.]|$)",
            name,
            re.IGNORECASE,
        ):
            return name

    return None


def pick_asset(names, flavor):
    """
    Возвращает:

        main_asset, cudart_asset

    Выбираются только Windows x64 ZIP.
    """

    if not names:
        return None, None

    zip_names = [
        name
        for name in names
        if name.lower().endswith(".zip")
        and "-bin-win-" in name.lower()
        and name.lower().endswith("-x64.zip")
    ]

    if not zip_names:
        return None, None

    def first_match(pattern):
        regex = re.compile(
            pattern,
            re.IGNORECASE,
        )

        for name in zip_names:
            if name.lower().startswith(
                "cudart-"
            ):
                continue

            if regex.search(name):
                return name

        return None

    flavor = (flavor or "cpu").lower()

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if flavor.startswith("cuda"):
        cuda_version = None

        match = re.fullmatch(
            r"cuda-cu([0-9]+\.[0-9]+)",
            flavor,
            re.IGNORECASE,
        )

        if match:
            cuda_version = match.group(1)

        main_asset = None

        if cuda_version:
            main_asset = first_match(
                rf"-bin-win-cuda-(?:cu)?"
                rf"{re.escape(cuda_version)}"
                rf"-x64\.zip$"
            )

        if not main_asset:
            candidates = []

            for name in zip_names:
                if name.lower().startswith(
                    "cudart-"
                ):
                    continue

                if re.search(
                    r"-bin-win-cuda-"
                    r"(?:cu)?[0-9]+(?:\.[0-9]+)?"
                    r"-x64\.zip$",
                    name,
                    re.IGNORECASE,
                ):
                    candidates.append(name)

            def cuda_sort_key(name):
                version = (
                    cuda_version_from_name(name)
                    or "0"
                )

                try:
                    return tuple(
                        int(part)
                        for part in version.split(".")
                    )

                except ValueError:
                    return (0,)

            candidates.sort(
                key=cuda_sort_key,
                reverse=True,
            )

            if candidates:
                main_asset = candidates[0]

        if not main_asset:
            return None, None

        main_version = (
            cuda_version_from_name(
                main_asset
            )
        )

        cudart_asset = pick_cudart_asset(
            names,
            main_version,
        )

        return main_asset, cudart_asset

    # --------------------------------------------------------
    # Vulkan
    # --------------------------------------------------------

    if flavor == "vulkan":
        main_asset = first_match(
            r"-bin-win-vulkan-x64\.zip$"
        )

        if main_asset:
            return main_asset, None

        return None, None

    # --------------------------------------------------------
    # HIP / ROCm
    # --------------------------------------------------------

    if flavor in (
        "hip",
        "hip-radeon",
    ):
        main_asset = first_match(
            r"-bin-win-hip.*-x64\.zip$"
        )

        if main_asset:
            return main_asset, None

        main_asset = first_match(
            r"-bin-win-rocm.*-x64\.zip$"
        )

        if main_asset:
            return main_asset, None

        return None, None

    # --------------------------------------------------------
    # SYCL
    # --------------------------------------------------------

    if flavor == "sycl":
        main_asset = first_match(
            r"-bin-win-sycl.*-x64\.zip$"
        )

        if main_asset:
            return main_asset, None

        return None, None

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    if flavor == "cpu":
        main_asset = first_match(
            r"-bin-win-cpu.*-x64\.zip$"
        )

        if main_asset:
            return main_asset, None

        return None, None

    return None, None


# ============================================================
# GITHUB API
# ============================================================

def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers=UA,
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        content = response.read()

    import json

    return json.loads(
        content.decode(
            "utf-8",
            errors="replace",
        )
    )


def published_timestamp(value: str):
    if not value:
        return 0

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).timestamp()

    except Exception:
        return 0


def find_latest_release(flavor):
    """
    ВАЖНО:

    prerelease намеренно не пропускается.
    Rolling-релизы llama.cpp часто имеют этот флаг.
    """

    try:
        releases = fetch_json(
            API_RELEASES
        )

    except Exception as exc:
        return {
            "ok": False,
            "error": (
                f"GitHub API недоступен: {exc}"
            ),
        }

    if not isinstance(releases, list):
        return {
            "ok": False,
            "error": (
                "GitHub API вернул неожиданный ответ."
            ),
        }

    candidates = []
    checked = []

    for release in releases:
        if not isinstance(release, dict):
            continue

        # Пропускаем только черновики.
        # prerelease оставляем.
        if release.get("draft"):
            continue

        tag = (
            release.get("tag_name") or ""
        ).strip()

        if not tag:
            continue

        checked.append(tag)

        assets = release.get("assets") or []
        assets_by_name = {}

        for asset in assets:
            if not isinstance(asset, dict):
                continue

            name = (
                asset.get("name") or ""
            ).strip()

            if name:
                assets_by_name[name] = asset

        names = list(
            assets_by_name.keys()
        )

        main_asset, cudart_asset = (
            pick_asset(
                names,
                flavor,
            )
        )

        if not main_asset:
            continue

        main_url = (
            assets_by_name[main_asset]
            .get("browser_download_url")
        )

        if not main_url:
            continue

        cudart_url = None

        if cudart_asset:
            cudart_url = (
                assets_by_name
                .get(cudart_asset, {})
                .get("browser_download_url")
            )

        candidates.append({
            "ok": True,
            "tag": tag,
            "build": get_build_from_tag(
                tag
            ),
            "published": (
                release.get("published_at")
                or ""
            ),
            "prerelease": bool(
                release.get("prerelease")
            ),
            "asset": main_asset,
            "asset_url": main_url,
            "cudart": cudart_asset,
            "cudart_url": cudart_url,
        })

    if not candidates:
        return {
            "ok": False,
            "error": (
                f"В просмотренных {len(checked)} "
                f"релизах нет Windows x64-сборки "
                f"для '{flavor}'."
            ),
            "checked": checked[:20],
        }

    # Выбираем максимальный build.
    # При равном build используется дата релиза.
    candidates.sort(
        key=lambda item: (
            item.get("build")
            if item.get("build") is not None
            else -1,
            published_timestamp(
                item.get("published") or ""
            ),
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# СРАВНЕНИЕ ВЕРСИЙ
# ============================================================

def compare_current_with_release(
    current,
    release,
):
    """
    Главное правило:

        текущий build >= доступный build
        => обновление не требуется

    Backend здесь намеренно не участвует.
    Версия является главным критерием.
    """

    if not current:
        return {
            "up_to_date": False,
            "can_update": True,
            "reason": (
                "Текущую версию EXE определить "
                "не удалось. Доступно обновление."
            ),
        }

    current_build = current.get(
        "build"
    )

    release_build = release.get(
        "build"
    )

    if (
        current_build is not None
        and release_build is not None
    ):
        if current_build >= release_build:
            return {
                "up_to_date": True,
                "can_update": False,
                "reason": (
                    "Обновление не требуется: "
                    f"установлена {current['display']}, "
                    f"доступен релиз "
                    f"{release['tag']}."
                ),
            }

        return {
            "up_to_date": False,
            "can_update": True,
            "reason": (
                "Доступно обновление: "
                f"установлена {current['display']}, "
                f"доступен более новый релиз "
                f"{release['tag']}."
            ),
        }

    if current.get("tag") == release.get(
        "tag"
    ):
        return {
            "up_to_date": True,
            "can_update": False,
            "reason": (
                "Обновление не требуется: "
                f"уже установлена версия "
                f"{release.get('tag')}."
            ),
        }

    return {
        "up_to_date": False,
        "can_update": True,
        "reason": (
            "Формат версии не позволяет надежно "
            "сравнить сборки. Доступна полная установка."
        ),
    }


# ============================================================
# ФАЙЛЫ И АРХИВЫ
# ============================================================

def is_process_running(exe_name: str):
    return_code, output = run_capture(
        [
            "tasklist",
            "/FI",
            f"IMAGENAME eq {exe_name}",
            "/NH",
        ],
        timeout=15,
    )

    if return_code != 0:
        return False

    return exe_name.lower() in output.lower()


def download_file(url, destination: Path):
    request = urllib.request.Request(
        url,
        headers=UA,
    )

    with urllib.request.urlopen(
        request,
        timeout=900,
    ) as response, open(
        destination,
        "wb",
    ) as output:
        shutil.copyfileobj(
            response,
            output,
            length=1024 * 512,
        )


def extract_zip_safe(
    zip_path: Path,
    output_dir: Path,
):
    """
    Распаковка с защитой от Zip Slip.
    """

    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            target = (
                output_dir / member
            ).resolve()

            try:
                target.relative_to(
                    output_root
                )

            except ValueError:
                raise RuntimeError(
                    f"Небезопасный путь в ZIP: "
                    f"{member}"
                )

        archive.extractall(
            output_dir
        )


def find_archive_bin_dir(
    extracted_dir: Path,
):
    for path in extracted_dir.rglob(
        "llama-server.exe"
    ):
        if path.is_file():
            return path.parent

    return None


def collect_runtime_files(folder: Path):
    """
    Собирает старые EXE/DLL.
    Остальные пользовательские файлы
    не затрагиваются.
    """

    result = []

    try:
        for path in folder.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() in (
                ".exe",
                ".dll",
            ):
                result.append(path)

    except OSError:
        pass

    return result


# ============================================================
# ОБНОВЛЕНИЕ
# ============================================================

def update_llama():
    folder = get_saved_folder()

    if not folder:
        return {
            "success": False,
            "error": (
                "llama_path.txt пуст или содержит "
                "несуществующий путь."
            ),
        }

    server = find_server_exe(folder)

    if not server:
        return {
            "success": False,
            "error": (
                f"llama-server.exe не найден в {folder}"
            ),
        }

    if is_process_running(
        "llama-server.exe"
    ):
        return {
            "success": False,
            "error": (
                "llama-server.exe запущен. "
                "Закройте его перед обновлением."
            ),
        }

    gpu = detect_gpu()
    flavor = gpu["flavor"]

    current, current_raw = (
        get_server_version(server)
    )

    current_backend, backend_sources = (
        detect_backend_details(folder)
    )

    release = find_latest_release(
        flavor
    )

    if not release.get("ok"):
        return {
            "success": False,
            "error": release.get(
                "error",
                "Подходящий релиз не найден.",
            ),
            "gpu": gpu,
            "current": current,
            "backend": current_backend,
            "backend_sources": backend_sources,
        }

    comparison = (
        compare_current_with_release(
            current,
            release,
        )
    )

    # Если build совпадает или текущий build новее,
    # ничего не переустанавливаем.
    if comparison["up_to_date"]:
        return {
            "success": True,
            "updated": False,
            "status": comparison["reason"],
            "current": current,
            "backend": current_backend,
            "backend_sources": backend_sources,
            "release": {
                "tag": release.get("tag"),
                "build": release.get("build"),
                "asset": release.get("asset"),
            },
            "gpu": gpu,
            "path": str(server.parent),
        }

    temp_dir = None
    backup_dir = None

    files_to_install = {}
    new_names = []
    old_names = []

    try:
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="llama_manager_",
                dir=str(BASE_DIR),
            )
        )

        main_zip = temp_dir / (
            "llama_main.zip"
        )

        download_file(
            release["asset_url"],
            main_zip,
        )

        if not zipfile.is_zipfile(
            main_zip
        ):
            raise RuntimeError(
                "Скачанный основной файл "
                "не является ZIP."
            )

        extracted_main = (
            temp_dir / "main"
        )

        extracted_main.mkdir()

        extract_zip_safe(
            main_zip,
            extracted_main,
        )

        source_dir = find_archive_bin_dir(
            extracted_main
        )

        if not source_dir:
            raise RuntimeError(
                "В ZIP не найден "
                "llama-server.exe."
            )

        # Основные EXE/DLL из архива.
        for source in source_dir.iterdir():
            if source.is_file():
                files_to_install[
                    source.name
                ] = source

        # Для CUDA может понадобиться отдельный cudart ZIP.
        if release.get("cudart_url"):
            cudart_zip = temp_dir / (
                "cudart.zip"
            )

            download_file(
                release["cudart_url"],
                cudart_zip,
            )

            if not zipfile.is_zipfile(
                cudart_zip
            ):
                raise RuntimeError(
                    "Скачанный cudart ZIP "
                    "поврежден."
                )

            extracted_cudart = (
                temp_dir / "cudart"
            )

            extracted_cudart.mkdir()

            extract_zip_safe(
                cudart_zip,
                extracted_cudart,
            )

            for dll in extracted_cudart.rglob(
                "*.dll"
            ):
                if dll.is_file():
                    files_to_install[
                        dll.name
                    ] = dll

        if not files_to_install:
            raise RuntimeError(
                "В архиве нет файлов "
                "для установки."
            )

        installed_names_lower = {
            name.lower()
            for name in files_to_install
        }

        if "llama-server.exe" not in (
            installed_names_lower
        ):
            raise RuntimeError(
                "В наборе файлов отсутствует "
                "llama-server.exe."
            )

        # Резервная копия старого runtime.
        backup_dir = Path(
            tempfile.mkdtemp(
                prefix=".llama_backup_",
                dir=str(server.parent),
            )
        )

        old_runtime_files = (
            collect_runtime_files(
                server.parent
            )
        )

        for old_file in old_runtime_files:
            backup_file = (
                backup_dir / old_file.name
            )

            shutil.copy2(
                old_file,
                backup_file,
            )

            old_names.append(
                old_file.name
            )

        # Удаляем только старые EXE/DLL.
        # GGUF, JSON, TXT и прочие файлы
        # пользователя не удаляются.
        for old_file in old_runtime_files:
            old_file.unlink()

        # Копируем новый runtime.
        for name, source in files_to_install.items():
            destination = (
                server.parent / name
            )

            new_names.append(name)

            shutil.copy2(
                source,
                destination,
            )

        installed_server = find_server_exe(
            server.parent
        )

        if not installed_server:
            raise RuntimeError(
                "После установки "
                "llama-server.exe не найден."
            )

        installed_version, installed_raw = (
            get_server_version(
                installed_server
            )
        )

        if not installed_version:
            raise RuntimeError(
                "После установки не удалось "
                "прочитать версию "
                "llama-server.exe.\n"
                f"Вывод:\n"
                f"{installed_raw[:700]}"
            )

        installed_backend, installed_sources = (
            detect_backend_details(
                server.parent
            )
        )

        release_build = release.get(
            "build"
        )

        if (
            release_build is not None
            and installed_version["build"]
            < release_build
        ):
            raise RuntimeError(
                "После установки build ниже "
                "ожидаемого: "
                f"{installed_version['display']}"
            )

        return {
            "success": True,
            "updated": True,
            "status": (
                "Runtime полностью заменен. "
                "Установлена подходящая сборка."
            ),
            "reason": comparison["reason"],
            "previous": current,
            "current": installed_version,
            "backend": installed_backend,
            "backend_sources": installed_sources,
            "release": {
                "tag": release.get("tag"),
                "build": release.get("build"),
                "asset": release.get("asset"),
            },
            "gpu": gpu,
            "files": len(files_to_install),
            "path": str(server.parent),
        }

    except Exception as exc:
        # Удаляем новые файлы неудачной установки.
        for name in new_names:
            try:
                (
                    server.parent / name
                ).unlink(
                    missing_ok=True
                )

            except OSError:
                pass

        # Восстанавливаем старые EXE/DLL.
        if backup_dir:
            for name in old_names:
                backup_file = (
                    backup_dir / name
                )

                if not backup_file.is_file():
                    continue

                try:
                    shutil.copy2(
                        backup_file,
                        server.parent / name,
                    )

                except OSError:
                    pass

        return {
            "success": False,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            "path": str(server.parent),
        }

    finally:
        if temp_dir:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

        if backup_dir:
            shutil.rmtree(
                backup_dir,
                ignore_errors=True,
            )


# ============================================================
# ДИАГНОСТИКА
# ============================================================

def get_info():
    folder = get_saved_folder()

    if not folder:
        return {
            "error": (
                "llama_path.txt пуст или содержит "
                "несуществующий путь."
            ),
            "Обновление доступно": False,
            "Обновление не требуется": False,
        }

    server = find_server_exe(folder)

    if not server:
        return {
            "error": (
                f"llama-server.exe не найден в {folder}"
            ),
            "Обновление доступно": False,
            "Обновление не требуется": False,
        }

    current, raw = (
        get_server_version(server)
    )

    gpu = detect_gpu()

    backend, backend_sources = (
        detect_backend_details(folder)
    )

    result = {
        "Путь": str(folder),
        "EXE": str(server),
        "Версия": (
            current["display"]
            if current
            else f"неизвестно ({raw[:250]})"
        ),
        "Распознанный build": (
            current["build"]
            if current
            else "—"
        ),
        "Commit": (
            current["commit"]
            if current
            else "—"
        ),
        "GPU": gpu["description"],
        "Производитель GPU": gpu["vendor"],
        "Рекомендуемая сборка": gpu["flavor"],
        "Backend текущего EXE": backend,
        "Источники определения backend": (
            backend_sources
        ),
        "Ожидаемый backend": expected_backend(
            gpu["flavor"]
        ),
        "Backend совпадает": backend_matches(
            gpu["flavor"],
            backend,
        ),
    }

    release = find_latest_release(
        gpu["flavor"]
    )

    if not release.get("ok"):
        result["Ошибка поиска релиза"] = (
            release.get(
                "error",
                "Релиз не найден.",
            )
        )

        result["Проверенные релизы"] = (
            release.get(
                "checked",
                [],
            )
        )

        result["Обновление доступно"] = False
        result["Обновление не требуется"] = False

        return result

    result["Последний подходящий релиз"] = {
        "tag": release.get("tag"),
        "build": release.get("build"),
        "asset": release.get("asset"),
        "prerelease": release.get(
            "prerelease"
        ),
    }

    comparison = (
        compare_current_with_release(
            current,
            release,
        )
    )

    result["Сравнение"] = (
        comparison["reason"]
    )

    result["Обновление доступно"] = (
        comparison["can_update"]
    )

    result["Обновление не требуется"] = (
        comparison["up_to_date"]
    )

    return result


# ============================================================
# WEB UI
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Llama Manager</title>

<style>
:root {
    --bg: #10131a;
    --panel: #191e28;
    --panel-light: #202735;
    --text: #e7ecf8;
    --muted: #9ca8bd;
    --accent: #7f66f4;
    --accent-hover: #9784ff;
    --green: #67d391;
    --red: #ff7777;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    padding: 28px;
    color: var(--text);
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(127, 102, 244, 0.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 90% 85%,
            rgba(62, 198, 157, 0.10),
            transparent 30%
        ),
        var(--bg);
    font-family: "Segoe UI", sans-serif;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
}

header {
    margin-bottom: 22px;
}

h1 {
    margin: 0 0 8px;
    color: #ffffff;
    font-size: 32px;
}

.subtitle {
    color: var(--muted);
}

.layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: 20px;
}

.panel {
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    background: rgba(25, 30, 40, 0.88);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.2);
}

.panel h2 {
    margin-top: 0;
    font-size: 18px;
}

pre {
    min-height: 500px;
    margin: 0;
    padding: 16px;
    overflow: auto;
    color: #b9c6dc;
    background: #0b0d12;
    border-radius: 10px;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: Consolas, monospace;
    font-size: 13px;
    line-height: 1.5;
}

button {
    width: 100%;
    margin-bottom: 10px;
    padding: 13px 16px;
    color: #ffffff;
    border: 0;
    border-radius: 10px;
    background: var(--accent);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition:
        background 0.2s ease,
        transform 0.2s ease,
        opacity 0.2s ease;
}

button:hover:not(:disabled) {
    background: var(--accent-hover);
    transform: translateY(-1px);
}

button:disabled {
    opacity: 0.42;
    cursor: not-allowed;
}

#bu:not(:disabled) {
    box-shadow: 0 0 0 1px rgba(127, 102, 244, 0.3),
        0 8px 24px rgba(127, 102, 244, 0.22);
}

.status {
    margin-top: 20px;
    padding: 14px;
    color: var(--muted);
    border-radius: 10px;
    background: var(--panel-light);
    font-size: 13px;
    line-height: 1.5;
}

.status strong {
    color: var(--green);
}

.log {
    height: 220px;
    min-height: 0;
    color: #72e59b;
}

@media (max-width: 900px) {
    body {
        padding: 16px;
    }

    .layout {
        grid-template-columns: 1fr;
    }

    pre {
        min-height: 360px;
    }
}
</style>
</head>

<body>

<div class="container">

    <header>
        <h1>Llama Manager</h1>
        <div class="subtitle">
            Определение GPU, backend и умное обновление llama.cpp
        </div>
    </header>

    <div class="layout">

        <section class="panel">
            <h2>Диагностика</h2>
            <pre id="diag">Загрузка...</pre>
        </section>

        <aside class="panel">
            <h2>Управление</h2>

            <button id="bd" onclick="runDiag()">
                Проверить систему
            </button>

            <button id="bu" onclick="runUpdate()" disabled>
                Обновление не требуется
            </button>

            <div class="status" id="status">
                Выполняется начальная проверка...
            </div>
        </aside>

    </div>

    <section class="panel" style="margin-top: 20px;">
        <h2>Лог выполнения</h2>
        <pre class="log" id="log"></pre>
    </section>

</div>

<script>
let interfaceBusy = false;
let updateAllowed = false;

function writeLog(text) {
    const element = document.getElementById("log");

    element.textContent += (text || "") + "\\n";
    element.scrollTop = element.scrollHeight;
}

function setStatus(text, success) {
    const element = document.getElementById("status");

    if (success) {
        element.innerHTML = "<strong>" + text + "</strong>";
    } else {
        element.textContent = text;
    }
}

function updateButtonState() {
    const button = document.getElementById("bu");

    button.disabled = (
        interfaceBusy
        || !updateAllowed
    );

    if (interfaceBusy) {
        button.textContent = "Проверка...";
        return;
    }

    if (updateAllowed) {
        button.textContent = "Обновить llama.cpp";
    } else {
        button.textContent = "Обновление не требуется";
    }
}

function setBusy(value) {
    interfaceBusy = value;

    document.getElementById("bd").disabled = value;

    updateButtonState();
}

function showDiagnostic(data) {
    document.getElementById("diag").textContent =
        JSON.stringify(data, null, 2);

    const info = data.info || data;

    updateAllowed = (
        info["Обновление доступно"] === true
    );

    const noUpdate = (
        info["Обновление не требуется"] === true
    );

    if (noUpdate) {
        setStatus(
            "Установлена актуальная версия.",
            true
        );
    } else if (updateAllowed) {
        setStatus(
            "Доступна новая версия llama.cpp.",
            false
        );
    } else if (info["Ошибка поиска релиза"]) {
        setStatus(
            "Не удалось определить доступный релиз.",
            false
        );
    } else {
        setStatus(
            "Состояние не определено.",
            false
        );
    }

    updateButtonState();
}

async function refreshDiagnostic() {
    const response = await fetch(
        "/api/diagnose"
    );

    const data = await response.json();

    showDiagnostic(data);

    return data;
}

async function runDiag() {
    setBusy(true);
    writeLog("> Проверка системы...");

    try {
        await refreshDiagnostic();

        writeLog(
            "> Диагностика завершена"
        );
    } catch (error) {
        writeLog(
            "Ошибка диагностики: " + error
        );

        setStatus(
            "Ошибка диагностики.",
            false
        );
    } finally {
        setBusy(false);
    }
}

async function runUpdate() {
    if (!updateAllowed) {
        writeLog(
            "> Обновление не требуется."
        );
        return;
    }

    setBusy(true);
    writeLog(
        "> Поиск последней Windows x64-сборки..."
    );

    try {
        const response = await fetch(
            "/api/update"
        );

        const data = await response.json();

        writeLog(
            JSON.stringify(data, null, 2)
        );

        await refreshDiagnostic();

    } catch (error) {
        writeLog(
            "Ошибка обновления: " + error
        );

        setStatus(
            "Ошибка обновления.",
            false
        );

    } finally {
        setBusy(false);
    }
}

runDiag();
</script>

</body>
</html>
"""


# ============================================================
# API
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PAGE


@app.get("/api/diagnose")
def diagnose():
    return get_info()


@app.get("/api/update")
def update():
    return update_llama()


# ============================================================
# ЗАПУСК WEB-МЕНЕДЖЕРА
# ============================================================

if __name__ == "__main__":
    print(
        "[PYTHON] Llama Manager:"
        f" http://{MANAGER_HOST}:{MANAGER_PORT}"
    )

    import uvicorn

    uvicorn.run(
        app,
        host=MANAGER_HOST,
        port=MANAGER_PORT,
    )
