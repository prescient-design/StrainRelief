from strain_relief.constants import (
    CHARGE_COL_NAME,
    CHARGE_KEY,
    ID_COL_NAME,
    MOL_COL_NAME,
    MOL_KEY,
    SPIN_COL_NAME,
    SPIN_KEY,
)


def test_constants():
    """Test that constants are of type str."""
    assert isinstance(ID_COL_NAME, str)
    assert isinstance(MOL_COL_NAME, str)
    assert isinstance(CHARGE_COL_NAME, str)
    assert isinstance(SPIN_COL_NAME, str)
    assert isinstance(MOL_KEY, str)
    assert isinstance(CHARGE_KEY, str)
    assert isinstance(SPIN_KEY, str)
