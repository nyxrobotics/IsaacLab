from __future__ import annotations

from pathlib import Path

import yaml


def export_io_descriptors(
    descriptors: dict,
    export_dir: str | Path,
    filename: str = "IO_descriptors.yaml",
) -> Path:
    """Write IO descriptors to a YAML file."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    file_path = export_path / filename
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(descriptors, f, sort_keys=False, allow_unicode=True)

    return file_path