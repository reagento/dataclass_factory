from collections import namedtuple
from typing import NamedTuple, TypeVar, Generic, Tuple

from dataclass_factory_30.feature_requirement import has_protocol
from dataclass_factory_30.type_tools import is_named_tuple_class, is_protocol, is_user_defined_generic


class NTParent(NamedTuple):
    a: int
    b: int


class NTChild(NTParent):
    c: int


DynNTParent = namedtuple('DynNTParent', 'a, b')


class DynNTChild(DynNTParent):
    c: int


def test_is_named_tuple_class():
    assert is_named_tuple_class(NTParent)
    assert is_named_tuple_class(NTChild)
    assert is_named_tuple_class(DynNTParent)
    assert is_named_tuple_class(DynNTChild)


@has_protocol
def test_is_protocol():
    from typing import Protocol, runtime_checkable
    from typing import SupportsInt

    class Proto(Protocol):
        def foo(self) -> bool:
            pass

    @runtime_checkable
    class RtProto(Protocol):
        def foo(self) -> bool:
            pass

    class ImplProto:
        def foo(self) -> bool:
            pass

    class InheritedImplProto(Proto):
        def foo(self) -> bool:
            pass

    class InheritedImplRtProto(RtProto):
        def foo(self) -> bool:
            pass

    assert not is_protocol(Protocol)
    assert is_protocol(Proto)
    assert is_protocol(RtProto)
    assert is_protocol(SupportsInt)

    assert not is_protocol(InheritedImplProto)
    assert not is_protocol(InheritedImplRtProto)

    assert not is_protocol(ImplProto)
    assert not is_protocol(int)
    assert not is_protocol(type)
    assert not is_protocol(object)

    assert not is_protocol(15)
    assert not is_protocol('15')

    class ExtProto(Proto, Protocol):
        def bar(self):
            pass

    assert is_protocol(ExtProto)


T = TypeVar('T')


class Gen(Generic[T]):
    pass


class GenChildImplicit(Gen):
    pass


class GenChildExplicit(Gen[int]):
    pass


class GenChildExplicitTypeVar(Gen[T]):
    pass


V = TypeVar('V')


class GenGen(Gen[int], Generic[T]):
    pass


def test_is_user_defined_generic():
    assert is_user_defined_generic(Gen)
    assert is_user_defined_generic(Gen[V])
    assert not is_user_defined_generic(Gen[int])

    assert not is_user_defined_generic(GenChildImplicit)
    assert not is_user_defined_generic(GenChildExplicit)

    assert is_user_defined_generic(GenChildExplicitTypeVar)
    assert is_user_defined_generic(GenChildExplicitTypeVar[V])
    assert not is_user_defined_generic(GenChildExplicitTypeVar[int])

    assert is_user_defined_generic(GenGen)
    assert is_user_defined_generic(GenGen[V])
    assert not is_user_defined_generic(GenGen[int])

    assert not is_user_defined_generic(Tuple)
    assert not is_user_defined_generic(Tuple[V])
    assert not is_user_defined_generic(Tuple[int])
