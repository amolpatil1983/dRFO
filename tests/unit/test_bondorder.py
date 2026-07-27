import pytest

from drfo.interpolation.bondorder import bond_length_from_order, bond_order


def test_bond_order_at_reference_length_is_one():
    assert bond_order(1.0, r0=1.0, b=2.0) == pytest.approx(1.0)


def test_bond_order_decays_to_zero_at_twice_reference_length():
    assert bond_order(2.0, r0=1.0, b=2.0) == pytest.approx(0.0, abs=1e-12)


def test_bond_order_zero_beyond_twice_reference_length():
    assert bond_order(3.0, r0=1.0, b=2.0) == 0.0


def test_bond_order_half_at_133_percent_reference_for_b2():
    # Paper: b=2 gives a bond length 133% of reference at order 0.5.
    r = 1.33
    assert bond_order(r, r0=1.0, b=2.0) == pytest.approx(0.5, abs=1e-2)


def test_round_trip_order_to_length_and_back():
    r0 = 1.4
    for order in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]:
        r = bond_length_from_order(order, r0)
        recovered = bond_order(r, r0)
        assert recovered == pytest.approx(order, abs=1e-8)


def test_bond_length_from_order_rejects_negative_order():
    with pytest.raises(ValueError):
        bond_length_from_order(-0.1, r0=1.4)
