
"""
PyTorch Structured Data Pipeline Loader.
Stacks input arrays matching LNO config: [state_t, phi, flux, noise, dissipation, instability]
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class IonTransportDataset(Dataset):
    def __init__(self, root_dir: str):
        """Loads physical array configurations saved across processing splits."""
        self.state_t = np.load(f"{root_dir}/state_t.npy")
        self.state_t1 = np.load(f"{root_dir}/state_t1.npy")
        self.phi = np.load(f"{root_dir}/phi.npy")
        self.flux = np.load(f"{root_dir}/flux.npy")
        self.noise = np.load(f"{root_dir}/noise.npy")
        self.dissipation = np.load(f"{root_dir}/dissipation.npy")
        self.entropy = np.load(f"{root_dir}/entropy.npy")
        self.instability = np.load(f"{root_dir}/instability.npy")

    def __len__(self) -> int:
        return len(self.state_t)

    def __getitem__(self, idx: int):
        # Multichannel structural stacking along axis=0 matching operator input dimension layouts
        x = np.stack([
            self.state_t[idx],
            self.phi[idx],
            self.flux[idx],
            self.noise[idx],
            self.dissipation[idx],
            self.instability[idx]
        ], axis=0)
        
        y = self.state_t1[idx]
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
