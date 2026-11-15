#!/usr/bin/env bash
set -euo pipefail
WORK=${VTR_WORK:?set VTR_WORK to a work folder outside the repository}
PY=${PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$WORK/official" "$WORK/weights/mobileone"

echo "== 1. official repositories (repro/official/repos.txt)"
while read -r name url commit; do
  [ -z "${name:-}" ] && continue
  if [ ! -d "$WORK/official/$name/.git" ]; then git clone -q "$url" "$WORK/official/$name"; fi
  git -C "$WORK/official/$name" fetch -q --depth 1 origin "$commit" 2>/dev/null || true
  git -C "$WORK/official/$name" checkout -q "$commit"
  echo "$name $(git -C "$WORK/official/$name" rev-parse --short HEAD)"
done < "$REPO/repro/official/repos.txt"

echo "== 2. base environment (timm 1.0.30, scikit-learn for ATC, our reimplementations)"
VENV_FLAGS=""
[ -n "${SKIP_TORCH:-}" ] && VENV_FLAGS="--system-site-packages"
[ -x "$WORK/venv_base/bin/python" ] || "$PY" -m venv $VENV_FLAGS "$WORK/venv_base"
B="$WORK/venv_base/bin/python"
$B -m pip install -q --upgrade pip
if [ -z "${SKIP_TORCH:-}" ]; then
  if [ -n "${TORCH_INDEX_URL:-}" ]; then $B -m pip install -q torch torchvision --index-url "$TORCH_INDEX_URL"
  else $B -m pip install -q torch torchvision; fi
fi
$B -m pip install -q "timm==1.0.30" scikit-learn scipy pyyaml einops torchprofile pillow numpy tqdm

echo "== 3. one environment per older timm version"
BASE_SITE=$($B -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for pair in "venv_timm04 0.4.12" "venv_timm067 0.6.7" "venv_timm092 0.9.2"; do
  set -- $pair
  E="$WORK/$1"
  [ -x "$E/bin/python" ] || "$PY" -m venv $VENV_FLAGS "$E"
  E_SITE=$("$E/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  echo "$BASE_SITE" > "$E_SITE/zz_venv_base.pth"
  "$E/bin/python" -m pip install -q --no-deps "timm==$2"
  echo "$1: $("$E/bin/python" -c 'import timm, torch; print("timm", timm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())')"
done
ln -sfn "$WORK/venv_base" "$WORK/venv_atc"

echo "== 4. weights"
$B "$REPO/repro/save_weights.py" --out "$WORK/weights"
M="$WORK/weights/mobileone/mobileone_s0.pth.tar"
if [ ! -f "$M" ]; then
  curl -sSL -o "$M" https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0.pth.tar
fi
echo "74080c64f108a15d9c73129f5ae8f8315981c8dc31e111e74b3ae36c7aea2033  $M" | sha256sum -c -

echo "== done. Use: export VTR_WORK=$WORK"
