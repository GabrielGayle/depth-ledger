"""Digital bas-relief from a cooked depth / look plate.

Based on Weyrich et al., "Digital Bas-Relief from 3D Scenes" (SIGGRAPH 2007),
with a vectorized adaptation of the MIT-licensed reference implementation:
https://github.com/wingated/digital-bas-relief
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

DEFAULT_RELIEF_GC_ALPHA = 10.0
DEFAULT_RELIEF_INVERT = False
DEFAULT_RELIEF_MAX_SIDE = 512
RELIEF_MAX_SIDE_CHOICES = (384, 512, 768, 1024)


def _to_gray01(img: np.ndarray) -> np.ndarray:
    """HxW float64 in [0, 1] from RGB or gray uint8/float plate."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        # Rec. 601 luminance
        rgb = arr[..., :3].astype(np.float64)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        gray = arr.astype(np.float64)
        if gray.max() > 1.5:
            gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0)


def _grad_compress(gx: np.ndarray, gy: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Weyrich gradient compression: keep direction, compress magnitude with log."""
    mags = np.sqrt(gx * gx + gy * gy + 1e-4)
    # φ(|∇|) = (1/α) log(1 + α|∇|)
    phi = (1.0 / alpha) * np.log1p(alpha * mags)
    scale = phi / mags
    return gx * scale, gy * scale


def _build_laplacian(rows: int, cols: int) -> sparse.csr_matrix:
    """5-point discrete Laplacian; Dirichlet zero on the image border."""
    n = rows * cols
    idx = np.arange(n, dtype=np.int64).reshape(rows, cols)
    border = np.zeros((rows, cols), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    r_idx: list[int] = []
    c_idx: list[int] = []
    vals: list[float] = []

    b = idx[border].ravel()
    r_idx.extend(b.tolist())
    c_idx.extend(b.tolist())
    vals.extend([1.0] * int(b.size))

    src = idx[1:-1, 1:-1]
    r_idx.extend(src.ravel().tolist())
    c_idx.extend(src.ravel().tolist())
    vals.extend([4.0] * int(src.size))

    neighbors = (
        idx[1:-1, 0:-2],  # left
        idx[1:-1, 2:],  # right
        idx[0:-2, 1:-1],  # up
        idx[2:, 1:-1],  # down
    )
    flat_src = src.ravel()
    for neigh in neighbors:
        r_idx.extend(flat_src.tolist())
        c_idx.extend(neigh.ravel().tolist())
        vals.extend([-1.0] * int(flat_src.size))

    return sparse.coo_matrix(
        (vals, (r_idx, c_idx)), shape=(n, n), dtype=np.float64
    ).tocsr()


def _integrate_poisson(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Reconstruct height from compressed gradients via Poisson (∇²h = div G)."""
    rows, cols = gx.shape
    div = np.zeros((rows, cols), dtype=np.float64)
    div[:, 1:-1] += gx[:, 1:-1] - gx[:, :-2]
    div[1:-1, :] += gy[1:-1, :] - gy[:-2, :]

    div[0, :] = 0.0
    div[-1, :] = 0.0
    div[:, 0] = 0.0
    div[:, -1] = 0.0

    A = _build_laplacian(rows, cols)
    b = div.ravel()
    n = rows * cols
    try:
        sol, info = cg(A, b, rtol=1e-5, maxiter=max(400, n // 50))
    except TypeError:
        # Older SciPy used `tol` instead of `rtol`.
        sol, info = cg(A, b, tol=1e-5, maxiter=max(400, n // 50))
    if info != 0:
        try:
            sol, _ = cg(A, b, x0=sol, rtol=1e-4, maxiter=max(800, n // 20))
        except TypeError:
            sol, _ = cg(A, b, x0=sol, tol=1e-4, maxiter=max(800, n // 20))
    return sol.reshape(rows, cols)


def _normalize01(hf: np.ndarray) -> np.ndarray:
    lo = float(np.min(hf))
    hi = float(np.max(hf))
    if hi - lo < 1e-12:
        return np.zeros_like(hf, dtype=np.float64)
    return (hf - lo) / (hi - lo)


def render_bas_relief(
    plate: np.ndarray,
    *,
    gc_alpha: float = DEFAULT_RELIEF_GC_ALPHA,
    invert: bool = DEFAULT_RELIEF_INVERT,
    max_side: int = DEFAULT_RELIEF_MAX_SIDE,
) -> np.ndarray:
    """Turn a look / depth plate into a smooth grayscale bas-relief (HxWx3 uint8).

    Parameters
    ----------
    plate:
        Incoming cooked plate (RGB or gray).
    gc_alpha:
        Gradient compression strength (paper α). Higher = flatter relief.
    invert:
        If True, treat bright pixels as far (raw depth-map convention).
        Default False matches Depth Ledger looks (bright = near / high).
    max_side:
        Longest edge for the Poisson solve. Result is resized back to ``plate``.
    """
    gray = _to_gray01(plate)
    hf = (1.0 - gray) if invert else gray

    src_h, src_w = hf.shape
    side = int(max(32, max_side))
    scale = 1.0
    work = hf
    if max(src_h, src_w) > side:
        scale = side / float(max(src_h, src_w))
        new_w = max(8, int(round(src_w * scale)))
        new_h = max(8, int(round(src_h * scale)))
        work = cv2.resize(hf, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rows, cols = work.shape
    # Pin border like the reference implementation.
    work = work.copy()
    work[0, :] = 0.0
    work[-1, :] = 0.0
    work[:, 0] = 0.0
    work[:, -1] = 0.0

    # Forward differences (ofs=0 in the reference).
    gx = np.zeros((rows, cols), dtype=np.float64)
    gy = np.zeros((rows, cols), dtype=np.float64)
    gx[:, 1:-1] = work[:, 1:-1] - work[:, :-2]
    gy[1:-1, :] = work[1:-1, :] - work[:-2, :]

    alpha = float(max(0.1, gc_alpha))
    cgx, cgy = _grad_compress(gx, gy, alpha)
    relief = _integrate_poisson(cgx, cgy)
    relief = _normalize01(relief)

    if scale != 1.0:
        relief = cv2.resize(relief, (src_w, src_h), interpolation=cv2.INTER_CUBIC)
        relief = _normalize01(relief)

    u8 = np.clip(np.round(relief * 255.0), 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def relief_params_from_data(data: Optional[dict]) -> dict:
    """Normalize Relief node data dict to render kwargs."""
    d = dict(data or {})
    max_side = int(d.get("max_side", DEFAULT_RELIEF_MAX_SIDE))
    if max_side not in RELIEF_MAX_SIDE_CHOICES:
        # Snap to nearest allowed
        max_side = min(RELIEF_MAX_SIDE_CHOICES, key=lambda s: abs(s - max_side))
    return {
        "gc_alpha": float(d.get("gc_alpha", DEFAULT_RELIEF_GC_ALPHA)),
        "invert": bool(d.get("invert", DEFAULT_RELIEF_INVERT)),
        "max_side": int(max_side),
    }
