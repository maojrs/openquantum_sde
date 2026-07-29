import h5py
import json
import numpy as np
from pathlib import Path


# =========================
# Trajectory I/O (HDF5)
# =========================

def save_trajectory(fname, output_dir, time_array, traj, traj_current, simid=None):
    fname = Path(fname).with_suffix(".h5")
    filename = output_dir / fname
    with h5py.File(filename, 'w') as f:
        f.create_dataset('time', data=time_array)
        f.create_dataset('traj', data=traj, compression='gzip')
        f.create_dataset('traj_current', data=traj_current, compression='gzip')

        # Minimal metadata
        f.attrs['sim_id'] = simid


def load_trajectory(fname, output_dir):
    fname = Path(fname).with_suffix(".h5")
    filename = output_dir / fname
    with h5py.File(filename, 'r') as f:
        time_array = f['time'][:]
        traj = f['traj'][:]
        traj_current = f['traj_current'][:]

    return time_array, traj, traj_current


def load_trajectory_chunk(fname, output_dir, chunk_size):
    fname = Path(fname).with_suffix(".h5")
    filename = output_dir / fname
    with h5py.File(filename, 'r') as f:
        chunk_traj = f["traj"][:chunk_size]
    return chunk_traj


def load_trajectory_chunk_random_start(fname, output_dir, chunk_size):
    fname = Path(fname).with_suffix(".h5")
    filename = output_dir / fname
    with h5py.File(filename, 'r') as f:
        traj = f["traj"]
        start = np.random.randint(0, traj.shape[0] - chunk_size)
        chunk_traj = traj[start:start+chunk_size]
        return chunk_traj


def load_trajectory_random_chunk(fname, output_dir, chunksize):
    fname = Path(fname).with_suffix(".h5")
    filename = output_dir / fname
    with h5py.File(filename, 'r') as f:
        traj = f["traj"]
        indices = np.random.choice(traj.shape[0], size=chunksize, replace=False)
        indices = np.sort(indices)
        chunk_traj = traj[indices]
    return chunk_traj


# =========================
# Parameters I/O (JSON)
# =========================

def save_params(fname, output_dir, params):
    filename = output_dir / fname
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)


def load_params(fname, output_dir):
    filename = output_dir / fname
    with open(filename, 'r') as f:
        return json.load(f)