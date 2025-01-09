import dataclasses

import pytest

from adaptix import Omittable, Omitted, Retort
from adaptix._internal.morphing.load_error import LoadError


@pytest.fixture
def retort():
    return Retort()


@dataclasses.dataclass
class WO:
    o: Omittable[str] = Omitted()


def test_valid_load_omittable(retort):
    assert retort.load({"o": "test"}, WO) == WO("test")


def test_load_omittable_wrong_type(retort):
    with pytest.raises(Exception):
        retort.load({"o":2}, WO)

def test_adaptix_sentinel_for_omittable(retort):
    with pytest.raises(LoadError):
        retort.load({"o": Omitted()}, WO)


def test_load_no_value(retort):
    with pytest.raises(LoadError):
        retort.load({}, WO)


def test_dump_valid_omittable(retort):
    assert retort.dump(WO(o="o"), WO) == {"o": "o"}


def test_dump_omitted(retort):
    with pytest.raises(ExceptionGroup):
        retort.dump(WO(), WO)


def  test_dump_omittable_wrong_type(retort):
    with pytest.raises(Exception):
        retort.dump(WO(1), WO)
