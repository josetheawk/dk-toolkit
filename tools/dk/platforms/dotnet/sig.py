"""Signature blobs (II.23.2) and custom attribute values (II.23.3) rendered as readable text."""

from __future__ import annotations

import struct
from typing import Callable

from .metadata import compressed_uint
from .pe import FormatError

PRIMITIVES = {
    0x01: "void", 0x02: "bool", 0x03: "char", 0x04: "sbyte", 0x05: "byte", 0x06: "short", 0x07: "ushort",
    0x08: "int", 0x09: "uint", 0x0A: "long", 0x0B: "ulong", 0x0C: "float", 0x0D: "double", 0x0E: "string",
    0x16: "typedref", 0x18: "nint", 0x19: "nuint", 0x1C: "object",
}
_PTR, _BYREF, _VALUETYPE, _CLASS, _VAR, _ARRAY, _GENERICINST = 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
_FNPTR, _SZARRAY, _MVAR, _CMOD_REQD, _CMOD_OPT, _SENTINEL, _PINNED = 0x1B, 0x1D, 0x1E, 0x1F, 0x20, 0x41, 0x45
_TYPEDEFORREF = ("TypeDef", "TypeRef", "TypeSpec")

TypeNamer = Callable[[tuple[str, int]], str]


class SigReader:
    def __init__(self, data: bytes, type_name: TypeNamer):
        self.data, self.at, self.type_name = data, 0, type_name

    def u8(self) -> int:
        if self.at >= len(self.data):
            raise FormatError("signature truncated")
        self.at += 1
        return self.data[self.at - 1]

    def uint(self) -> int:
        value, used = compressed_uint(self.data, self.at)
        self.at += used
        return value

    def peek(self) -> int:
        return self.data[self.at] if self.at < len(self.data) else -1

    def typedeforref(self) -> str:
        value = self.uint()
        tag = value & 3
        if tag > 2:
            raise FormatError("bad TypeDefOrRefEncoded tag")
        return self.type_name((_TYPEDEFORREF[tag], value >> 2))

    def type(self) -> str:
        mods = []
        while self.peek() in (_CMOD_REQD, _CMOD_OPT):
            kind = "modreq" if self.u8() == _CMOD_REQD else "modopt"
            mods.append(f" {kind}({self.typedeforref()})")
        t = self.u8()
        if t in PRIMITIVES:
            base = PRIMITIVES[t]
        elif t in (_CLASS, _VALUETYPE):
            base = self.typedeforref()
        elif t == _PTR:
            base = self.type() + "*"
        elif t == _BYREF:
            base = self.type() + "&"
        elif t == _SZARRAY:
            base = self.type() + "[]"
        elif t == _ARRAY:
            elem, rank = self.type(), self.uint()
            for _ in range(self.uint()):    # sizes
                self.uint()
            for _ in range(self.uint()):    # lower bounds (signed, same width)
                self.uint()
            base = f"{elem}[{',' * (rank - 1)}]"
        elif t == _GENERICINST:
            self.u8()                       # CLASS or VALUETYPE
            generic = self.typedeforref()
            args = [self.type() for _ in range(self.uint())]
            base = f"{generic.split('`')[0]}<{', '.join(args)}>"
        elif t == _VAR:
            base = f"!{self.uint()}"
        elif t == _MVAR:
            base = f"!!{self.uint()}"
        elif t == _FNPTR:
            base = "method " + self.method()
        elif t == _PINNED:
            base = self.type() + " pinned"
        elif t == _SENTINEL:
            base = "..."
        else:
            raise FormatError(f"unknown element type {t:#x}")
        return base + "".join(mods)

    def method(self, name: str = "") -> str:
        """MethodDefSig / MethodRefSig / StandAloneMethodSig → 'ret name(params)'."""
        conv = self.u8()
        generic = f"<{self.uint()}>" if conv & 0x10 else ""
        count = self.uint()
        ret = self.type()
        params = [self.type() for _ in range(count)]
        static = "" if conv & 0x20 else "static "
        return f"{static}{ret} {name}{generic}({', '.join(params)})"

    def method_parts(self) -> tuple[bool, str, list[str]]:
        """(has this, return type, parameter types)."""
        conv = self.u8()
        if conv & 0x10:
            self.uint()
        count = self.uint()
        ret = self.type()
        return bool(conv & 0x20), ret, [self.type() for _ in range(count)]


def field_type(blob: bytes, type_name: TypeNamer) -> str:
    r = SigReader(blob, type_name)
    if r.u8() != 0x06:
        raise FormatError("not a field signature")
    return r.type()


def property_type(blob: bytes, type_name: TypeNamer) -> str:
    r = SigReader(blob, type_name)
    if r.u8() & 0x0F != 0x08:
        raise FormatError("not a property signature")
    count = r.uint()
    ret = r.type()
    params = [r.type() for _ in range(count)]
    return f"{ret}[{', '.join(params)}]" if params else ret


def local_types(blob: bytes, type_name: TypeNamer) -> list[str]:
    r = SigReader(blob, type_name)
    if r.u8() != 0x07:
        raise FormatError("not a local variable signature")
    return [r.type() for _ in range(r.uint())]


def method_spec(blob: bytes, type_name: TypeNamer) -> str:
    r = SigReader(blob, type_name)
    if r.u8() != 0x0A:
        raise FormatError("not a method spec")
    return "<" + ", ".join(r.type() for _ in range(r.uint())) + ">"


def is_field_sig(blob: bytes) -> bool:
    return bool(blob) and blob[0] == 0x06


# Custom attribute values

_CA_FIXED = {0x02: "<?", 0x03: "<H", 0x04: "<b", 0x05: "<B", 0x06: "<h", 0x07: "<H", 0x08: "<i", 0x09: "<I",
             0x0A: "<q", 0x0B: "<Q", 0x0C: "<f", 0x0D: "<d"}


class _CaReader:
    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def take(self, fmt: str):
        (value,) = struct.unpack_from(fmt, self.data, self.at)
        self.at += struct.calcsize(fmt)
        return value

    def ser_string(self) -> str | None:
        if self.data[self.at] == 0xFF:
            self.at += 1
            return None
        length, used = compressed_uint(self.data, self.at)
        self.at += used + length
        return self.data[self.at - length:self.at].decode("utf-8", "replace")

    def value(self, code: int) -> object:
        """One value of element type `code` (enums are read as int32)."""
        if code == 0x03:
            return chr(self.take("<H"))
        if code in _CA_FIXED:
            return self.take(_CA_FIXED[code])
        if code in (0x0E, 0x50):                 # string, System.Type (by name)
            return self.ser_string()
        if code in (_CLASS, _VALUETYPE, 0x55):   # an enum whose underlying type isn't known here
            return self.take("<i")
        if code == 0x51:                         # boxed: the tag comes first
            return self.value(self.field_or_prop_type())
        raise FormatError(f"custom attribute type {code:#x} not decoded")

    def field_or_prop_type(self) -> int:
        code = self.take("<B")
        if code == _SZARRAY:
            self.take("<B")
            raise FormatError("array in a custom attribute not decoded")
        if code == 0x55:
            self.ser_string()
        return code


def custom_attribute_args(value: bytes, ctor_sig: bytes) -> list[str]:
    """Constructor arguments and named arguments of one attribute, rendered; raises FormatError."""
    if value[:2] != b"\x01\x00":
        raise FormatError("custom attribute without prolog")
    r = SigReader(ctor_sig, lambda ref: "?")
    conv = r.u8()
    if conv & 0x10:
        r.uint()
    count = r.uint()
    r.u8()                                       # return type (void)
    codes = []
    for _ in range(count):
        code = r.u8()
        if code in (_CLASS, _VALUETYPE):
            r.uint()                             # the type token is not needed to read an enum value
        elif code == _SZARRAY:
            raise FormatError("array parameter not decoded")
        codes.append(code)
    ca = _CaReader(value)
    ca.at = 2
    out = [repr(ca.value(code)) for code in codes]
    if ca.at + 2 <= len(value):
        for _ in range(ca.take("<H")):
            ca.take("<B")                        # FIELD (0x53) or PROPERTY (0x54)
            code = ca.field_or_prop_type()
            name = ca.ser_string()
            out.append(f"{name}={ca.value(code)!r}")
    return out
