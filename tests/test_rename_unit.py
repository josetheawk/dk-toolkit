from tools.dk.platforms.gc_mwcc.renamecmd import placeholder_pairs


def test_placeholder_pairs_map_members_to_mangled_names():
    known = {"fn_800C6634", "fn_800C6644", "vf1__Foo", "fn_800C6700__3FooFv"}
    compiled = ["fn_800C6634__Q22AI10HelicopterFfff", "fn_800C6644", "vf1__Foo",
                "fn_800C6700__3FooFv", "fn_80000000__3BarFv"]
    assert placeholder_pairs(compiled, known) == [("fn_800C6634", "fn_800C6634__Q22AI10HelicopterFfff")]


def test_c_uses_find_link_names_not_cpp_uses():
    from tools.dk.platforms.gc_mwcc.renamecmd import c_uses
    text = '''#include "x.h"
extern "C" void fn_80001000(int);            // C link name
extern "C" {
void fn_80002000(void);
}
// fn_80001000 in a comment is ignored
struct Foo { void fn_80003000(int); };
void caller(Foo* f) { f->fn_80003000(1); Foo::fn_80003000; fn_80001000__3FooFv(); }
asm void a() { bl fn_80003000
}
'''
    names = {"fn_80001000", "fn_80002000", "fn_80003000"}
    assert c_uses(text, names) == [("fn_80001000", 2), ("fn_80002000", 4), ("fn_80003000", 9)]
    assert c_uses("void fn_80003000(int);\n", names, c_file=True) == [("fn_80003000", 1)]


class _Fake:
    def __init__(self, sections, symbols, relocs):
        from tools.dk.platforms.gc_mwcc.elf import ElfSection, ElfSymbol
        self.sections = tuple([ElfSection(0, "", 0, 0, 0, 0, 0, b"")] +
                              [ElfSection(i + 1, n, 1, 0, 0, 0, 0, b"") for i, n in enumerate(sections)])
        # (name, value, size, type, bind, shndx)
        self.symbols = tuple(ElfSymbol(n, v, sz, t, b, idx) for n, v, sz, t, b, idx in symbols)
        self._relocs = relocs

    def relocations(self, index):
        return self._relocs.get(index, [])


def test_address_pairs_by_vtable_slot_and_offset():
    from tools.dk.platforms.gc_mwcc.elf import Reloc
    from tools.dk.platforms.gc_mwcc.renamecmd import address_pairs
    F, O, GLOBAL, WEAK = 2, 1, 1, 2
    ours = _Fake([".text", ".data"],
                 [("__dt__3FooFv", 0x00, 0x40, F, GLOBAL, 1),        # differs in size from the target's
                  ("@8@__dt__3FooFv", 0x40, 0x8, F, WEAK, 1),         # thunk: weak, still ours
                  ("vf1__3BarFv", 0x48, 0x10, F, WEAK, 1),            # inline base copy: never renamed
                  ("Helper__3FooFv", 0x58, 0x20, F, GLOBAL, 1),       # same offset and size
                  ("__vt__3Foo", 0, 0x18, O, GLOBAL, 2)],
                 {2: [Reloc(0x8, 1, "__dt__3FooFv", 0), Reloc(0xC, 1, "vf1__3BarFv", 0)]})
    target = _Fake([".text", ".data"],
                   [("fn_80001000", 0x00, 0x48, F, GLOBAL, 1), ("fn_80001048", 0x48, 0x8, F, GLOBAL, 1),
                    ("fn_80001050", 0x50, 0x8, F, GLOBAL, 1), ("fn_80001058", 0x58, 0x20, F, GLOBAL, 1),
                    ("fn_80002000", 0x80, 0x10, F, GLOBAL, 1),
                    ("__vt__3Foo", 0, 0x18, O, GLOBAL, 2)],
                   {2: [Reloc(0x8, 1, "fn_80001000", 0), Reloc(0xC, 1, "fn_80002000", 0),
                        Reloc(0x14, 1, "fn_80001048", 0)]})
    ours._relocs[2].append(Reloc(0x14, 1, "@8@__dt__3FooFv", 0))
    known = {"fn_80001000", "fn_80001048", "fn_80001050", "fn_80001058", "fn_80002000"}
    assert address_pairs(ours, target, known) == [
        ("fn_80001000", "__dt__3FooFv"), ("fn_80001048", "@8@__dt__3FooFv"), ("fn_80001058", "Helper__3FooFv")]
    assert address_pairs(ours, target, known, taken={"fn_80001058"}) == [
        ("fn_80001000", "__dt__3FooFv"), ("fn_80001048", "@8@__dt__3FooFv")]
