from strain_relief.constants import (
    ENERGY_PROPERTY_NAME,
    ID_COL_NAME,
    MOL_COL_NAME,
)


def test_constants():
    assert isinstance(ID_COL_NAME, str)
    assert isinstance(MOL_COL_NAME, str)
    assert isinstance(ENERGY_PROPERTY_NAME, str)
