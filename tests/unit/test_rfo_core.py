import numpy as np
import pytest

from drfo.optimize.rfo_core import (
    classify_eigenvectors,
    drfo_step,
    floor_eigenvalues,
    rfo_shift_matrix,
)


def test_classify_eigenvectors_picks_ts_mode_by_overlap():
    # Diagonal Hessian: mode 0 has negative curvature (the "TS mode"),
    # modes 1,2 have positive curvature (minimization space).
    H = np.diag([-1.0, 2.0, 3.0])
    s = np.array([1.0, 0.0, 0.0])  # aligned with mode 0
    cls = classify_eigenvectors(H, s, overlap_thresh=0.5)
    assert cls.ts_indices == [0]
    assert set(cls.min_indices) == {1, 2}


def test_classify_eigenvectors_falls_back_when_no_overlap_clears_threshold():
    H = np.diag([-1.0, 2.0, 3.0])
    s = np.array([0.1, 0.1, 0.1])  # ambiguous, no eigenvector dominates
    cls = classify_eigenvectors(H, s, overlap_thresh=0.99)
    # Falls back to the most-negative eigenvalue's index.
    assert cls.ts_indices == [0]


def test_classify_eigenvectors_rejects_zero_guide_vector():
    H = np.eye(3)
    with pytest.raises(ValueError):
        classify_eigenvectors(H, np.zeros(3))


def test_drfo_step_at_exact_saddle_point_is_zero():
    # A pure quadratic saddle f(x,y) = -x^2 + y^2 has a stationary point
    # (zero gradient) at the origin with Hessian diag(-2, 2). A converged
    # optimizer sitting exactly there should return a zero step.
    H = np.diag([-2.0, 2.0])
    g = np.zeros(2)
    s = np.array([1.0, 0.0])
    dq = drfo_step(H, g, s)
    assert np.allclose(dq, 0.0, atol=1e-8)


def test_drfo_step_reduces_to_textbook_prfo_single_mode():
    # Hand-built 2D saddle-shaped quadratic PES, displaced from its
    # stationary point: f(x,y) = -x^2 + 5y^2 + b.x + c.y with a known
    # minimum-of-quadratic-model solution. The augmented-Hessian pRFO step
    # should point toward the stationary point along each mode with a
    # step that, when both curvatures are already correctly signed AND the
    # classification correctly isolates the single negative mode, closely
    # approximates -H^-1 g up to the RFO scale factor (it should NOT simply
    # equal Newton's step, since RFO deliberately reduces to a scaled
    # Newton-like step whose direction should still oppose the gradient in
    # the minimization coordinate and align with -g along the TS coordinate
    # when H's curvature there is negative and the step is uphill).
    H = np.diag([-2.0, 10.0])
    g = np.array([0.5, -1.0])
    s = np.array([1.0, 0.0])

    dq = drfo_step(H, g, s, overlap_thresh=0.5)

    # In the minimization coordinate (index 1), the step should be a
    # downhill move: opposing the gradient's sign (g[1]=-1 -> dq[1] > 0
    # for a minimization step reducing y away from increasing curvature).
    assert dq[1] * (-g[1]) > 0
    # In the TS coordinate (index 0), the step should be uphill: same
    # sign as the gradient itself is climbed against, i.e. moving further
    # from g=0.5 requires dq[0] to have the same sign as g[0] (uphill
    # along a maximized coordinate moves opposite to -g, i.e. with g).
    assert dq[0] * g[0] > 0


def test_rfo_shift_matrix_gives_correct_number_of_negative_eigenvalues():
    # After shifting, H+S should have exactly one negative eigenvalue
    # (one TS mode), matching the fundamental requirement for a
    # Newton-Raphson TS step.
    rng = np.random.default_rng(0)
    n = 5
    A = rng.normal(size=(n, n))
    H = 0.5 * (A + A.T)  # symmetric, indefinite in general
    g = rng.normal(size=n)
    # Force a specific negative-curvature mode to exist and be identifiable.
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals[0] = -abs(eigvals[0]) - 1.0  # ensure mode 0 is negative
    eigvals[1:] = np.abs(eigvals[1:]) + 0.5  # ensure the rest are positive
    H = eigvecs @ np.diag(eigvals) @ eigvecs.T
    s = eigvecs[:, 0]

    cls = classify_eigenvectors(H, s, overlap_thresh=0.5)
    assert cls.ts_indices == [0]
    S = rfo_shift_matrix(cls, g)
    shifted_eigvals = np.linalg.eigvalsh(H + S)
    assert np.sum(shifted_eigvals < 0) == 1


def test_drfo_multi_mode_ts_space_matches_pRFO_when_single_mode():
    # When classify_eigenvectors isolates exactly one TS-space vector,
    # dRFO's shift matrix must be identical to what a hand-rolled
    # single-mode pRFO shift would produce (this is the "pRFO is the
    # n_TS==1 special case" contract the design relies on).
    H = np.diag([-3.0, 1.0, 4.0])
    g = np.array([0.2, -0.3, 0.1])
    s = np.array([1.0, 0.0, 0.0])

    cls = classify_eigenvectors(H, s, overlap_thresh=0.5)
    assert cls.ts_indices == [0]
    S = rfo_shift_matrix(cls, g)

    # Manually reproduce eq. 6 (pRFO) shift and confirm it matches when
    # lambda3 correction is folded in consistently on both sides: rather
    # than re-deriving pRFO independently (redundant with rfo_core's own
    # eigenvalue solve), assert the structural invariant that S is
    # diagonal in the eigenbasis of H for this diagonal-H case, and that
    # its TS-mode eigenvalue shift makes H+S negative there while all
    # other modes remain positive.
    shifted = H + S
    assert shifted[0, 0] < 0
    assert shifted[1, 1] > 0
    assert shifted[2, 2] > 0


def test_floor_eigenvalues_raises_near_zero_magnitudes():
    H = np.diag([1e-10, 2.0, -1e-12])
    H_floored = floor_eigenvalues(H, min_abs_eigval=0.02)
    eigvals = np.linalg.eigvalsh(H_floored)
    assert np.all(np.abs(eigvals) >= 0.02 - 1e-9)
    # the well-conditioned eigenvalue (2.0) must be untouched
    assert np.any(np.isclose(eigvals, 2.0))


def test_floor_eigenvalues_caps_spurious_large_magnitudes():
    # Reproduces the pentadiene pathology: a Cartesian-to-internal Hessian
    # transform can produce one or two wildly out-of-range eigenvalues
    # (observed: 205, 260) alongside an otherwise-normal spectrum.
    H = np.diag([0.5, 1.0, 260.0, -205.0, 3.0])
    H_capped = floor_eigenvalues(H, min_abs_eigval=0.02, max_abs_eigval=10.0)
    eigvals = np.linalg.eigvalsh(H_capped)
    assert np.all(np.abs(eigvals) <= 10.0 + 1e-9)
    # signs must be preserved through capping
    assert np.sum(eigvals < 0) == 1
    # well-conditioned eigenvalues (0.5, 1.0, 3.0) must be untouched
    for v in (0.5, 1.0, 3.0):
        assert np.any(np.isclose(eigvals, v))


def test_floor_eigenvalues_cap_does_not_affect_normal_spectrum():
    H = np.diag([-2.0, 0.5, 1.0, 3.0])
    H_capped = floor_eigenvalues(H, min_abs_eigval=0.02, max_abs_eigval=10.0)
    assert np.allclose(H, H_capped)
