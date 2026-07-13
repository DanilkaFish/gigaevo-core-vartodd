#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VARTODD_REPO_URL="${VARTODD_REPO_URL:-https://github.com/DanilkaFish/VarTodd.git}"
VARTODD_BRANCH="${VARTODD_BRANCH:-no-data-scripts}"
SOURCE_DIR="${VARTODD_SOURCE_DIR:-$ROOT_DIR/.vartodd-src}"
BUILD_DIR="${VARTODD_BUILD_DIR:-}"
BUILD_TYPE="${VARTODD_BUILD_TYPE:-Release}"
INSTALL_DIR="${VARTODD_INSTALL_DIR:-$ROOT_DIR/pyvartodd/Release}"
WITH_STUBS="${VARTODD_WITH_STUBS:-OFF}"
FETCHCONTENT_UPDATES_DISCONNECTED="${VARTODD_FETCHCONTENT_UPDATES_DISCONNECTED:-ON}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command '$1' was not found" >&2
        exit 1
    fi
}

require_command git
require_command cmake

if [ -n "${PYTHON:-}" ]; then
    PYTHON_EXECUTABLE="$(command -v "$PYTHON" 2>/dev/null || printf '%s' "$PYTHON")"
else
    PYTHON_EXECUTABLE="$(command -v python || command -v python3 || true)"
fi
if [ -z "$PYTHON_EXECUTABLE" ] || [ ! -x "$PYTHON_EXECUTABLE" ]; then
    echo "error: Python executable not found; set PYTHON=/path/to/python" >&2
    exit 1
fi
PYTHON_ABI="$("$PYTHON_EXECUTABLE" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="$SOURCE_DIR/build-gigaevo-py$PYTHON_ABI"
fi

if [ ! -d "$SOURCE_DIR/.git" ]; then
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone --branch "$VARTODD_BRANCH" --depth "${VARTODD_CLONE_DEPTH:-1}" \
        "$VARTODD_REPO_URL" "$SOURCE_DIR"
else
    git -C "$SOURCE_DIR" checkout "$VARTODD_BRANCH"
    if [ "${VARTODD_UPDATE:-0}" = "1" ]; then
        git -C "$SOURCE_DIR" fetch origin "$VARTODD_BRANCH"
        git -C "$SOURCE_DIR" merge --ff-only "origin/$VARTODD_BRANCH"
    fi
fi

cmake_args=(
    -S "$SOURCE_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    -DBUILD_DEMONSTRATION=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_PYTHON_BINDINGS=ON
    -DWITH_PY_STUBS="$WITH_STUBS"
    -DFETCHCONTENT_UPDATES_DISCONNECTED="$FETCHCONTENT_UPDATES_DISCONNECTED"
    -DPython_EXECUTABLE="$PYTHON_EXECUTABLE"
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE"
    -DPYBIND11_FINDPYTHON=ON
)

if [ -n "${VARTODD_CMAKE_GENERATOR:-}" ]; then
    cmake_args=(-G "$VARTODD_CMAKE_GENERATOR" "${cmake_args[@]}")
fi
if [ -n "${VARTODD_FETCHCONTENT_BASE_DIR:-}" ]; then
    cmake_args+=(-DFETCHCONTENT_BASE_DIR="$VARTODD_FETCHCONTENT_BASE_DIR")
fi

cmake "${cmake_args[@]}"

build_args=(--build "$BUILD_DIR" --config "$BUILD_TYPE" --target pyvartodd)
if [ -n "${VARTODD_BUILD_JOBS:-}" ]; then
    build_args+=(--parallel "$VARTODD_BUILD_JOBS")
else
    build_args+=(--parallel)
fi
cmake "${build_args[@]}"

shopt -s nullglob
extension_candidates=(
    "$SOURCE_DIR"/pyvartodd/"$BUILD_TYPE"/pyvartodd*.so
    "$SOURCE_DIR"/pyvartodd/pyvartodd*.so
    "$BUILD_DIR"/lib/"$BUILD_TYPE"/pyvartodd*.so
    "$BUILD_DIR"/lib/pyvartodd*.so
)
cnpy_candidates=(
)
stub_candidates=(
)
shopt -u nullglob

for candidate in \
    "$BUILD_DIR/lib/$BUILD_TYPE/libcnpy++.so" \
    "$BUILD_DIR/lib/libcnpy++.so" \
    "$SOURCE_DIR/build/lib/$BUILD_TYPE/libcnpy++.so" \
    "$SOURCE_DIR/build/lib/libcnpy++.so"; do
    if [ -f "$candidate" ]; then
        cnpy_candidates+=("$candidate")
    fi
done

for candidate in \
    "$SOURCE_DIR/pyvartodd/$BUILD_TYPE/pyvartodd.pyi" \
    "$SOURCE_DIR/pyvartodd/pyvartodd.pyi" \
    "$BUILD_DIR/lib/$BUILD_TYPE/pyvartodd.pyi" \
    "$BUILD_DIR/lib/pyvartodd.pyi"; do
    if [ -f "$candidate" ]; then
        stub_candidates+=("$candidate")
    fi
done

if [ "${#extension_candidates[@]}" -eq 0 ]; then
    echo "error: pyvartodd extension was not produced by the build" >&2
    exit 1
fi
if [ "${#cnpy_candidates[@]}" -eq 0 ]; then
    echo "error: libcnpy++.so was not produced by the build" >&2
    exit 1
fi

selected_extension="${extension_candidates[0]}"

mkdir -p "$INSTALL_DIR"
find "$INSTALL_DIR" -maxdepth 1 -type f \( \
    -name 'pyvartodd*.so' -o \
    -name 'pyvartodd.pyi' -o \
    -name 'libcnpy++.so' \
\) -delete

cp "$selected_extension" "$INSTALL_DIR/"
cp "${cnpy_candidates[0]}" "$INSTALL_DIR/"
if [ "${#stub_candidates[@]}" -gt 0 ]; then
    cp "${stub_candidates[0]}" "$INSTALL_DIR/"
fi

echo "Installed pyvartodd from $VARTODD_REPO_URL#$VARTODD_BRANCH"
echo "  source:  $SOURCE_DIR"
echo "  build:   $BUILD_DIR"
echo "  install: $INSTALL_DIR"
echo
echo "The VarTODD problem wrappers use this directory by default."
echo "For a custom location, set VARTODD_PYVARTODD_DIR=$INSTALL_DIR before running GigaEvo."
