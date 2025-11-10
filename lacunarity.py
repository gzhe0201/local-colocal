import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- Paste your lacunarity_3d() function here ---
from numpy.lib.stride_tricks import sliding_window_view

def lacunarity_3d(mask, box_sizes=None, normalize_fill=True):
    if mask.ndim != 3:
        raise ValueError("Input mask must be a 3D array (Z, Y, X).")

    mask = mask.astype(np.float32)
    fill_fraction = np.mean(mask)

    if normalize_fill:
        target_fill = 0.5
        if fill_fraction > 0:
            p = target_fill / fill_fraction
            p = min(p, 1.0)
            rng = np.random.default_rng(seed=42)
            mask = (mask * (rng.random(mask.shape) < p)).astype(np.float32)
        else:
            raise ValueError("Mask is empty; cannot normalize filling fraction.")

    if box_sizes is None:
        min_dim = min(mask.shape)
        box_sizes = [2**i for i in range(1, int(np.log2(min_dim//4)) + 1)]
    box_sizes = np.array(box_sizes, dtype=int)

    Lambda_vals = []

    for r in box_sizes:
        if any(dim < r for dim in mask.shape):
            continue

        patches = sliding_window_view(mask, (r, r, r))
        masses = patches.sum(axis=(-3, -2, -1)).ravel()
        mu = np.mean(masses)
        sigma2 = np.var(masses)
        Lambda = sigma2 / (mu ** 2) + 1 if mu > 0 else np.nan
        Lambda_vals.append(Lambda)

    Lambda_vals = np.array(Lambda_vals)
    Lambda_mean = np.nanmean(Lambda_vals)

    return {"r": box_sizes, "Lambda": Lambda_vals,
            "fill_fraction": fill_fraction, "Lambda_mean": Lambda_mean}


# --- Synthetic test mask generators ---

def uniform_cube(size=64):
    return np.ones((size, size, size), dtype=bool)

def random_points(size=64, p=0.05):
    return np.random.rand(size, size, size) < p

def clustered_blobs(size=64, n_blobs=20, radius=5):
    mask = np.zeros((size, size, size), bool)
    zz, yy, xx = np.ogrid[:size, :size, :size]
    for _ in range(n_blobs):
        cz, cy, cx = np.random.randint(0, size, 3)
        blob = (zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 < radius**2
        mask |= blob
    return mask

def lattice(size=64, spacing=8):
    mask = np.zeros((size, size, size), bool)
    mask[::spacing, ::spacing, ::spacing] = 1
    return mask

"""def perlin_noise(size=64, sigma=4, threshold=0.7):
    noise = np.random.rand(size, size, size)
    smooth = gaussian_filter(noise, sigma=sigma)
    return smooth > threshold

def sierpinski_cube(n=3):
    base = np.ones((3,3,3), bool)
    base[1,1,1] = 0
    result = base.copy()
    for _ in range(n-1):
        result = np.kron(result, base)
    return result"""


# --- Run tests ---
patterns = {
    "Uniform cube": uniform_cube(),
    "Random points": random_points(),
    "Clustered blobs": clustered_blobs(),
    "Lattice": lattice(),
}

results = {}
for name, mask in patterns.items():
    res = lacunarity_3d(mask, box_sizes=[2,4,8,16,32], normalize_fill=True)
    results[name] = res
    print(f"{name:18s}  Fill={res['fill_fraction']:.3f}  Mean Λ={res['Lambda_mean']:.3f}")

# --- Plot ---
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(res["r"], res["Lambda"], marker="o", label=name)

plt.xscale("log")
plt.xlabel("Box size (r, voxels)")
plt.ylabel("Lacunarity Λ(r)")
plt.title("3D Lacunarity validation on synthetic patterns")
plt.legend()
plt.tight_layout()
plt.show()
