from pxr import Usd
from pathlib import Path
import os

_CANELE_DIR = os.path.dirname(__file__)
_CANELE_USD_PATH = os.path.join(_CANELE_DIR, "canele.usd")

# ★ exists() ＋ コロンを付ける
if not Path(_CANELE_USD_PATH).exists():
    raise FileNotFoundError(f"USD file not found: {_CANELE_USD_PATH}")

stage = Usd.Stage.Open(str(_CANELE_USD_PATH))
if stage is None:
    raise RuntimeError(f"Failed to open stage: {_CANELE_USD_PATH}")

removed = 0

for prim in stage.Traverse():
    # その prim が持っている全ての relationship を調べる
    for rel in prim.GetRelationships():
        name = rel.GetName()
        # "geometry:material:binding" や "material:binding" など、material binding 系を全部対象にする
        if name.endswith("material:binding"):
            rel.ClearTargets(True)          # 参照を消す
            prim.RemoveProperty(name)   # プロパティ自体を消す
            removed += 1

print(f"Removed {removed} material bindings.")
stage.GetRootLayer().Save()
print(f"Saved cleaned USD to: {_CANELE_USD_PATH}")
