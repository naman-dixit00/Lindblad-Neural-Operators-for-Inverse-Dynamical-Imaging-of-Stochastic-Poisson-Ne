"""
Master Dataset Pipeline Execution Orchestrator.
Generates structural regimes, processes distributions, handles splits and saves HDF5/NPY.
"""

import os
import json
import numpy as np
import h5py
import yaml

from data.pnp_simulator import PhysicsOperatorSimulator
from data.regime_configs import REGIMES
from data.validation import validate_sample, dataset_statistics

def main():
    # Load pipeline parameters master configurations file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config["infrastructure"]["seed"])
    simulator = PhysicsOperatorSimulator(config)
    trajectories_count = config["dataset"]["trajectories_per_regime"]
    
    all_state_t = []
    all_state_t1 = []
    all_phi = []
    all_flux = []
    all_noise = []
    all_diss = []
    all_gamma = []
    all_entropy = []
    all_instability = []
    
    print("=== Initiating Yuánlǐ AI High-Fidelity Physics Data Engine ===")
    for regime_name, parameters in REGIMES.items():
        print(f"Running Environment Phase Execution: [{regime_name}]")
        for _ in range(trajectories_count):
            st, st1, ph, fl, ns, ds, gm, ent, inst = simulator.compute_trajectory(
                noise_sigma=parameters["noise"],
                gamma=parameters["gamma"]
            )
            # Filter arrays through automated validation metrics checks
            if validate_sample(st[0], st1[0]):
                all_state_t.append(st)
                all_state_t1.append(st1)
                all_phi.append(ph)
                all_flux.append(fl)
                all_noise.append(ns)
                all_diss.append(ds)
                all_gamma.append(gm)
                all_entropy.append(ent)
                all_instability.append(inst)

    # Convert lists into continuous operational arrays
    X_state_t = np.concatenate(all_state_t, axis=0)
    Y_state_t1 = np.concatenate(all_state_t1, axis=0)
    X_phi = np.concatenate(all_phi, axis=0)
    X_flux = np.concatenate(all_flux, axis=0)
    X_noise = np.concatenate(all_noise, axis=0)
    X_diss = np.concatenate(all_diss, axis=0)
    X_gamma = np.concatenate(all_gamma, axis=0)
    X_entropy = np.concatenate(all_entropy, axis=0)
    X_instability = np.concatenate(all_instability, axis=0)

    # Perform unified array shuffling across coordinated blocks
    total_samples = len(X_state_t)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    X_state_t = X_state_t[indices]
    Y_state_t1 = Y_state_t1[indices]
    X_phi = X_phi[indices]
    X_flux = X_flux[indices]
    X_noise = X_noise[indices]
    X_diss = X_diss[indices]
    X_gamma = X_gamma[indices]
    X_entropy = X_entropy[indices]
    X_instability = X_instability[indices]

    # Calculate train / val / test indices segment distributions
    train_end = int(config["dataset"]["train_ratio"] * total_samples)
    val_end = train_end + int(config["dataset"]["val_ratio"] * total_samples)
    
    splits = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_samples)
    }

    # Save target specific arrays across layout paths
    for split_name, (start, end) in splits.items():
        split_path = f"dataset/{split_name}"
        os.makedirs(split_path, exist_ok=True)
        
        np.save(f"{split_path}/state_t.npy", X_state_t[start:end])
        np.save(f"{split_path}/state_t1.npy", Y_state_t1[start:end])
        np.save(f"{split_path}/phi.npy", X_phi[start:end])
        np.save(f"{split_path}/flux.npy", X_flux[start:end])
        np.save(f"{split_path}/noise.npy", X_noise[start:end])
        np.save(f"{split_path}/dissipation.npy", X_diss[start:end])
        np.save(f"{split_path}/gamma.npy", X_gamma[start:end])
        np.save(f"{split_path}/entropy.npy", X_entropy[start:end])
        np.save(f"{split_path}/instability.npy", X_instability[start:end])

    # Export high-end academic HDF5 unified file storage 
    print("Exporting production-scale master HDF5 layout dataset...")
    h5_path = "dataset/operator_dataset.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("state_t", data=X_state_t, compression="gzip")
        hf.create_dataset("state_t1", data=Y_state_t1, compression="gzip")
        hf.create_dataset("phi", data=X_phi, compression="gzip")
        hf.create_dataset("flux", data=X_flux, compression="gzip")
        hf.create_dataset("noise", data=X_noise, compression="gzip")
        hf.create_dataset("dissipation", data=X_diss, compression="gzip")
        hf.create_dataset("gamma", data=X_gamma, compression="gzip")
        hf.create_dataset("entropy", data=X_entropy, compression="gzip")
        hf.create_dataset("instability", data=X_instability, compression="gzip")

    # Generate metadata documents under metadata/ directories
    os.makedirs("metadata", exist_ok=True)
    os.makedirs("dataset/metadata", exist_ok=True)
    meta_path = "metadata"
    
    with open(f"{meta_path}/regimes.json", "w") as fj:
        json.dump(REGIMES, fj, indent=2)
        
    with open(f"{meta_path}/dataset_config.json", "w") as fj:
        json.dump(config["dataset"], fj, indent=2)
        
    stats = {
        "total_samples": total_samples,
        "density_stats": dataset_statistics(X_state_t),
        "flux_stats": dataset_statistics(X_flux),
        "entropy_range": dataset_statistics(X_entropy)
    }
    with open(f"{meta_path}/statistics.json", "w") as fj:
        json.dump(stats, fj, indent=2)
        
    print(f"=== Process Completed. Data vectors compiled successfully inside dataset/ splits. Total samples: {total_samples} ===")

if __name__ == "__main__":
    main()
