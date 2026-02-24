"""
split_czi_timesteps.py

Splits a multi-timestep .czi file into separate .czi files, one per timepoint.

Dependencies:
    pip install pylibCZIrw aicspylibczi numpy

Usage:
    python split_czi_timesteps.py input.czi
    python split_czi_timesteps.py input.czi --output_dir ./split_output
    python split_czi_timesteps.py input.czi --timesteps 0 5 10  # specific timepoints only
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from aicspylibczi import CziFile
from pylibCZIrw import czi as pylibczi


def get_czi_info(czi_path: str) -> dict:
    """Read CZI metadata and return dimension info."""
    czi = CziFile(czi_path)
    dims = czi.get_dims_shape()  # list of dicts, one per scene
    print(f"File: {czi_path}")
    print(f"Dimensions: {czi.dims}")
    print(f"Shape info: {dims}")
    return {"czi": czi, "dims": dims}


def split_czi_by_timestep(
    input_path: str,
    output_dir: str = None,
    timesteps: list[int] = None,
):
    """
    Split a multi-timestep CZI into individual CZI files.

    Parameters
    ----------
    input_path : str
        Path to the input .czi file.
    output_dir : str, optional
        Directory for output files. Defaults to same directory as input.
    timesteps : list of int, optional
        Specific timesteps to extract. Defaults to all.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_split"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input CZI
    czi = CziFile(str(input_path))
    dims_string = czi.dims  # e.g. 'STCZYX'
    dims_shape = czi.get_dims_shape()  # list of dicts per scene

    if "T" not in dims_string:
        print("No time dimension found in this CZI file. Nothing to split.")
        return

    # Determine number of scenes and timepoints
    n_scenes = len(dims_shape)
    n_timepoints = dims_shape[0].get("T", (0, 1))
    # dims_shape entries are like {'T': (start, size), 'C': (start, size), ...}
    t_start, t_size = n_timepoints
    print(f"Found {t_size} timepoints (starting at index {t_start}), {n_scenes} scene(s)")

    if timesteps is not None:
        selected_ts = [t for t in timesteps if t_start <= t < t_start + t_size]
        if len(selected_ts) < len(timesteps):
            skipped = set(timesteps) - set(selected_ts)
            print(f"Warning: timesteps {skipped} are out of range and will be skipped.")
    else:
        selected_ts = list(range(t_start, t_start + t_size))

    print(f"Extracting {len(selected_ts)} timepoint(s)...")

    for t_idx in selected_ts:
        out_name = f"{input_path.stem}_T{t_idx:04d}.czi"
        out_path = output_dir / out_name

        print(f"  Writing timepoint {t_idx} -> {out_path}")

        with pylibczi.create_czi(str(out_path)) as czw:
            for s_idx in range(n_scenes):
                scene_dims = dims_shape[s_idx]

                # Determine channel range
                c_start, c_size = scene_dims.get("C", (0, 1))
                # Determine Z range
                z_start, z_size = scene_dims.get("Z", (0, 1))

                for c in range(c_start, c_start + c_size):
                    for z in range(z_start, z_start + z_size):
                        # Build the read kwargs
                        read_kwargs = {"T": t_idx, "C": c, "Z": z}
                        if n_scenes > 1:
                            read_kwargs["S"] = s_idx

                        # Read the 2D plane
                        img, shape_info = czi.read_image(**read_kwargs)

                        # img may have extra leading singleton dims — squeeze to 2D
                        plane = np.squeeze(img)
                        if plane.ndim == 3:
                            # Likely (1, H, W) or (H, W, 1); take the 2D slice
                            plane = plane.reshape(plane.shape[-2], plane.shape[-1])

                        # Write kwargs: remap T->0 so each file starts at T=0
                        write_kwargs = {"C": c, "Z": z}
                        if n_scenes > 1:
                            write_kwargs["S"] = s_idx

                        czw.write(
                            data=plane,
                            plane=write_kwargs,
                        )

    print(f"\nDone! {len(selected_ts)} file(s) written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a multi-timestep .czi file into separate .czi files per timepoint."
    )
    parser.add_argument("input", help="Path to the input .czi file")
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Output directory (default: <input_stem>_split/ next to input file)",
    )
    parser.add_argument(
        "--timesteps", "-t",
        nargs="+",
        type=int,
        default=None,
        help="Specific timestep indices to extract (default: all)",
    )
    args = parser.parse_args()
    split_czi_by_timestep(args.input, args.output_dir, args.timesteps)


if __name__ == "__main__":
    main()
