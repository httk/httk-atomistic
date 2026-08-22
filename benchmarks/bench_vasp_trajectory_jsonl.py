"""Benchmark conversion of the workspace Al trajectory to JSONL."""

import tempfile
import time
from pathlib import Path

import httk.core

from httk.atomistic import VASPTrajectory

AL_300K = Path(__file__).resolve().parents[2] / "electronic-structure-example-data" / "MD" / "VASP" / "Al_300K"


def main() -> int:
    """Run the conversion benchmark when its workspace fixture is available."""
    if not AL_300K.is_dir():
        print(f"skipped: workspace fixture unavailable: {AL_300K}")
        return 0
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "Al_300K.traj.jsonl"
        started = time.perf_counter()
        httk.core.save(VASPTrajectory(AL_300K), output)
        loaded = httk.core.load(output)
        frames = loaded.nframes
        elapsed = time.perf_counter() - started
        assert frames == 10000
    print(f"frames={frames} seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
