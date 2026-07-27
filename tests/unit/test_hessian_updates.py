import numpy as np

from drfo.hessian.updates import bfgs_update, bofill_update


def test_bfgs_satisfies_secant_condition():
    rng = np.random.default_rng(0)
    n = 4
    H = np.eye(n)
    dx = rng.normal(size=n)
    dg = H @ dx + rng.normal(scale=0.1, size=n)  # ensure dx.dg > 0 typically
    # force a positive curvature condition
    if dx @ dg <= 0:
        dg = -dg
    H_new = bfgs_update(H, dx, dg)
    assert np.allclose(H_new @ dx, dg, atol=1e-8)


def test_bfgs_skips_update_on_curvature_condition_failure():
    H = np.eye(3)
    dx = np.array([1.0, 0.0, 0.0])
    dg = np.array([-1.0, 0.0, 0.0])  # dx.dg < 0, violates BFGS curvature condition
    H_new = bfgs_update(H, dx, dg)
    assert np.allclose(H_new, H)


def test_bfgs_stays_symmetric_positive_definite():
    rng = np.random.default_rng(1)
    H = np.eye(3)
    for _ in range(5):
        dx = rng.normal(size=3)
        dg = H @ dx + 0.05 * rng.normal(size=3)
        if dx @ dg <= 0:
            continue
        H = bfgs_update(H, dx, dg)
    assert np.allclose(H, H.T)
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(eigvals > 0)


def test_bofill_satisfies_secant_condition():
    rng = np.random.default_rng(2)
    n = 4
    H = np.diag([1.0, -0.5, 2.0, 0.3])  # indefinite, as expected during TS search
    dx = rng.normal(size=n)
    dg = rng.normal(size=n)
    H_new = bofill_update(H, dx, dg)
    assert np.allclose(H_new @ dx, dg, atol=1e-8)


def test_bofill_stays_symmetric():
    H = np.diag([1.0, -0.5, 2.0])
    dx = np.array([0.1, 0.2, -0.1])
    dg = np.array([0.05, -0.3, 0.2])
    H_new = bofill_update(H, dx, dg)
    assert np.allclose(H_new, H_new.T)


def test_bofill_can_introduce_negative_curvature():
    # Starting from a positive-definite H, a secant pair pointing "the
    # wrong way" (dg much smaller than what H@dx predicts, opposite trend)
    # should let Bofill reduce curvature along dx, unlike BFGS which would
    # refuse the update outright when dx.dg < 0.
    H = np.eye(2) * 2.0
    dx = np.array([1.0, 0.0])
    dg = np.array([-1.0, 0.0])
    H_new = bofill_update(H, dx, dg)
    assert np.allclose(H_new @ dx, dg, atol=1e-8)
    assert H_new[0, 0] < 0


def test_bofill_skips_on_degenerate_step():
    H = np.eye(3)
    dx = np.zeros(3)
    dg = np.array([1.0, 0.0, 0.0])
    H_new = bofill_update(H, dx, dg)
    assert np.allclose(H_new, H)
