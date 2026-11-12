"""The HMMWV of the benchmark (plan task 5.4, docs/hmmwv.md) driven at 10 mph
over a half-cylinder bump, played back in Rerun (plan task 6.5).

The run is made in C++, where the car and its tyres are built:

    scripts\\build.ps1 -Target mbd_hmmwv
    build\\bench\\mbd_hmmwv.exe record <folder>           # or: record <folder> bushed

which writes <folder>/hmmwv.json (the model, without its tyres: they are
forces written as code, which a model file cannot hold and the playback does
not need) and <folder>/hmmwv_run.csv (the coordinates every 10 ms for 5 s).
Then:

    python python/examples/replay_hmmwv.py <folder>              # opens the viewer
    python python/examples/replay_hmmwv.py <folder> hmmwv.rrd    # or saves a recording

The plots show the coordinates and, as a channel, the chassis frame's height.
"""

import sys
from pathlib import Path

import multibody as mbd


def main(folder, path=None, name="hmmwv"):
    folder = Path(folder)
    car = mbd.load_model(folder / f"{name}.json")
    run = mbd.read_results(folder / f"{name}_run.csv")
    # The chassis is body 1, on a free joint: its coordinates q[0] to q[2]
    # are its frame's position.
    height = run[["q[2]"]].rename(columns={"q[2]": "chassis frame height [m]"})
    mbd.replay(car, run, name="HMMWV", path=path, channels=height)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
