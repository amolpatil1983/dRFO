import numpy as np

from drfo.hessian.deltab import delta_b_vector, project_transition_vector
from drfo.hessian.initial import build_initial_ts_hessian
from drfo.internal.coordinates import Bend, InternalCoordinateSystem, Stretch


def test_delta_b_vector_signs():
    ics = InternalCoordinateSystem(
        coords=[Stretch(0, 1), Stretch(0, 2), Bend(1, 0, 2)], natoms=3,
    )
    s = delta_b_vector(ics, breaking=[(0, 1)], forming=[(0, 2)])
    assert s[0] == -1.0  # breaking
    assert s[1] == 1.0   # forming
    assert s[2] == 0.0   # bend untouched


def test_project_transition_vector_full_rank_B_is_identity_like():
    # If B has full row rank (n_int <= 3N and independent rows), B @ B+ is
    # the identity on that row space, so projecting s onto it should
    # return s unchanged (up to numerical precision).
    B = np.array([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, -1.0, 0.0]])
    s = np.array([1.0, -1.0])
    s_NR = project_transition_vector(B, s)
    assert np.allclose(s_NR, s, atol=1e-8)


def test_project_transition_vector_removes_unachievable_component():
    # A redundant coordinate row (duplicate of the first) means s can have
    # components that project onto a subspace consistent with B's rank;
    # here we just check the result stays finite and its norm doesn't
    # exceed the original (a projector is non-expansive).
    B = np.array([[1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],  # redundant with row 0
                  [0.0, 1.0, 0.0]])
    s = np.array([1.0, -1.0, 0.5])
    s_NR = project_transition_vector(B, s)
    assert np.all(np.isfinite(s_NR))
    assert np.linalg.norm(s_NR) <= np.linalg.norm(s) + 1e-8


def test_build_initial_ts_hessian_flips_curvature_along_s():
    n = 3
    H_tilde = np.eye(n) * 2.0  # positive definite, as from BFGS relaxation
    s = np.array([1.0, 0.0, 0.0])
    H0 = build_initial_ts_hessian(H_tilde, s, flip_scale=1.5)

    rayleigh_before = float(s @ H_tilde @ s) / float(s @ s)
    rayleigh_after = float(s @ H0 @ s) / float(s @ s)
    assert rayleigh_before > 0
    assert rayleigh_after < 0  # curvature along s is now negative

    # orthogonal directions should be unaffected
    assert H0[1, 1] == H_tilde[1, 1]
    assert H0[2, 2] == H_tilde[2, 2]


def test_build_initial_ts_hessian_rejects_zero_guide_vector():
    import pytest
    H_tilde = np.eye(3)
    with pytest.raises(ValueError):
        build_initial_ts_hessian(H_tilde, np.zeros(3))
