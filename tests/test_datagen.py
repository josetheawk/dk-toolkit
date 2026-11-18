import unittest

from tools.dk.platforms.gc_mwcc import datagen
from tools.dk.platforms.gc_mwcc.dtk_symbols import Symbol


def sym(name, section, address, size, data=None, scope="global"):
    attrs = {"type": "object", "size": f"0x{size:X}", "scope": scope}
    if data:
        attrs["data"] = data
    return Symbol(name, section, address, attrs)


def draft(s, text, src="", here=("fn_a",), elsewhere=(), data_users=()):
    return datagen.draft(s, datagen.items(text), s.section, src, list(here), list(elsewhere), list(data_users),
                         s.attrs["scope"])


class ItemsTest(unittest.TestCase):
    def test_words_relocations_strings_and_skips(self):
        got = datagen.items('.obj x\n\t.4byte 0x3F800000\n\t.4byte fn_80401000\n\t.string "a\\"b"\n\t.skip 0x8\n.endobj x\n')
        self.assertEqual([(i.kind, i.value, i.size) for i in got],
                         [("word", 0x3F800000, 4), ("reloc", "fn_80401000", 4), ("string", b'a"b', 4), ("skip", None, 8)])


class DraftTest(unittest.TestCase):
    def test_compiler_data_is_left_alone(self):
        self.assertEqual(draft(sym("__vt__3Foo", ".data", 0, 12), ".4byte 0").status, "skip")
        self.assertEqual(draft(sym("jumptable_80401000", ".data", 0, 8), ".rel fn_a, .L_1").status, "skip")
        self.assertEqual(draft(sym("lbl_1", ".data", 0, 4, "string"), '.string "abc"').status, "skip")   # its code's literal
        self.assertEqual(draft(sym("lbl_2", ".sdata2", 0, 4, "float"), ".4byte 0x3F800000").status, "skip")

    def test_zero_filled_objects_take_the_externs_type_or_the_element_type(self):
        d = draft(sym("lbl_3", ".bss", 0x100, 0x280, "4byte"), ".skip 0x280", src="extern Foo lbl_3[20];\n")
        self.assertEqual((d.text, d.replaces), ("Foo lbl_3[20];", (0, 22)))
        self.assertEqual(draft(sym("lbl_4", ".sbss", 0, 1, "byte"), ".skip 0x1").text, "unsigned char lbl_4;")
        self.assertEqual(draft(sym("lbl_5", ".bss", 0, 0x10, "float", "local"), ".skip 0x10").text, "static float lbl_5[4];")

    def test_values_from_the_original_with_their_type(self):
        d = draft(sym("lbl_6", ".rodata", 0, 0x10, "4byte"), ".4byte 0xFFFFFFFB\n.4byte 0\n.4byte 5\n.4byte 0xA")
        self.assertEqual(d.text, "const int lbl_6[4] = {-5, 0, 5, 10};")
        d = draft(sym("lbl_7", ".data", 0, 8, "float"), ".4byte 0x3F800000\n.4byte 0x3DCCCCCD", elsewhere=("fn_b",))
        self.assertEqual(d.text, "float lbl_7[2] = {1.0f, 0.1f};")

    def test_a_table_of_functions(self):
        d = draft(sym("lbl_8", ".data", 0, 8), ".4byte fn_80401000\n.4byte fn_80402000")
        self.assertEqual(d.text, "void (*lbl_8[2])() = {fn_80401000, fn_80402000};")
        self.assertEqual(d.needs, ['extern "C" void fn_80401000();', 'extern "C" void fn_80402000();'])
        self.assertEqual(draft(sym("lbl_9", ".data", 0, 4), ".4byte foo__3BarFv").status, "todo")

    def test_no_evidence_is_a_todo_never_a_byte_dump(self):
        d = draft(sym("lbl_10", ".data", 0, 8), ".4byte 0x12345678\n.4byte 0x00000001", elsewhere=("fn_b",))
        self.assertEqual(d.status, "todo")
        self.assertIn("0x8 bytes with no type evidence", d.why)
        self.assertEqual(d.text, "")

    def test_an_already_defined_object(self):
        src = "static int lbl_11 = 1;\nvoid f() {\n    T* e = &lbl_12[3];\n}\n"
        self.assertTrue(datagen.defined_in(src, "lbl_11"))
        self.assertFalse(datagen.defined_in(src, "lbl_12"))


class CompilerDataTest(unittest.TestCase):
    """What gen --data must never define: data the compiler emits itself (UiDC's shapes)."""

    def test_a_class_name_string_is_rtti(self):
        s = sym("lbl_8078DE5C", ".sdata", 0, 8, "string")
        d = datagen.draft(s, datagen.items('.string "UiDC"'), ".sdata", "", [], ["fn_x"], [], "global",
                          class_names=frozenset({"UiDC", "XDC"}))
        self.assertEqual(d.status, "skip")
        self.assertIn("class name", d.why)

    def test_a_base_list_referenced_only_from_rtti(self):
        s = sym("lbl_8078DE64", ".sdata", 0, 8)
        d = draft(s, ".4byte lbl_8078DE5C\n.4byte 0", here=(), data_users=("__RTTI__4UiDC",))
        self.assertEqual(d.status, "skip")
        self.assertIn("referenced only from data", d.why)

    def test_strings_an_existing_table_uses(self):
        s = sym("lbl_80650000", ".data", 0, 8, "string")
        d = draft(s, '.string "FEF_BIG"', here=(), data_users=("lbl_FEFNames",))
        self.assertEqual(d.status, "skip")

    def test_a_local_arrays_initializer_template(self):
        s = sym("lbl_806394E0", ".rodata", 0, 0x10, "4byte")
        d = datagen.draft(s, datagen.items(".4byte 0xFFFFFFFB\n.4byte 0\n.4byte 5\n.4byte 0xA"), ".rodata", "",
                          ["fn_80505BF4"], [], [], "global", copied=True)
        self.assertEqual(d.status, "skip")
        self.assertIn("initializer", d.why)

    def test_a_table_of_strings_writes_them_as_literals(self):
        s = sym("lbl_80650100", ".data", 0, 8)
        d = datagen.draft(s, datagen.items(".4byte lbl_a\n.4byte lbl_b"), ".data", "", ["fn_x"], [], [], "global",
                          strings={"lbl_a": b"one", "lbl_b": b"two"})
        self.assertEqual(d.text, 'const char* lbl_80650100[2] = {"one", "two"};')
        self.assertEqual(d.evidence, "2 relocations to strings, written as literals")


class IndexedTableTest(unittest.TestCase):
    def test_an_int_table_with_printable_bytes_is_not_strings(self):
        # snd/80591684 lbl_80672A00: effect-object sizes {0x20, 0x28, 0x58}, read with lwzx off a hoisted base
        text = ".4byte 0x00000020\n.4byte 0x00000028\n.4byte 0x00000058\n"
        s = sym("lbl_80672A00", ".data", 0, 0xC)
        d = draft(s, text)
        self.assertNotEqual(d.why, "string literals of the unit's code (pooled with their functions)")
        asm = "lis r3, lbl_80672A00@ha\naddi r31, r3, lbl_80672A00@l\nslwi r0, r4, 2\nlwzx r3, r31, r0\n"
        self.assertTrue(datagen.indexed_read("lbl_80672A00", [asm]))
        self.assertFalse(datagen.indexed_read("lbl_80672A00", ["lwz r3, lbl_80672A00@l(r3)\n"]))

    def test_real_string_blobs_still_are(self):
        self.assertTrue(datagen._string_blob(b"%svolumes.ini\0numgroup\0\0"))
        self.assertFalse(datagen._string_blob(b"\0\0\0 \0\0\0(\0\0\0X"))


class S1gCasesTest(unittest.TestCase):
    def test_a_local_static_and_its_guard(self):
        # 80407410 .sbss: float 1414 + guard byte 1418, both read by the function that owns them
        objs = [sym("lbl_80791414", ".sbss", 0x1414, 4, "float"), sym("lbl_80791418", ".sbss", 0x1418, 1, "byte"),
                sym("lbl_8079141C", ".sbss", 0x141C, 4, "4byte")]
        users = {"lbl_80791414": ["fn_a"], "lbl_80791418": ["fn_a"], "lbl_8079141C": ["fn_b"]}
        self.assertEqual(datagen.local_statics(objs, users), {"lbl_80791414", "lbl_80791418"})

    def test_a_template_copied_word_by_word_into_the_frame(self):
        # ElementModelPlatform lbl_80638CA0[28]: float bits copied through a pointer into the frame
        asm = ("lis r3, lbl_80638CA0@ha\naddi r4, r3, lbl_80638CA0@l\naddi r5, r1, 0x8\n"
               "lwz r0, 0x0(r4)\nstw r0, 0x0(r5)\n")
        self.assertTrue(datagen.copied_to_stack("lbl_80638CA0", [asm]))
        self.assertFalse(datagen.copied_to_stack("lbl_80638CA0", ["addi r4, r3, lbl_80638CA0@l\nlwz r0, 0x0(r4)\n"
                                                                 "stw r0, 0x10(r30)\n"]))

    def test_a_copy_loop_through_derived_pointers(self):
        # 80410974 lbl_80638F24[64]: subi r4, base, 4 / addi r5, r1, K, then lwzu / stwu in a ctr loop
        asm = ("addi r31, r3, lbl_80638F24@l\nli r0, 0x20\naddi r5, r1, 0xc\nsubi r4, r31, 0x4\nmtctr r0\n"
               ".L_1:\nlwzu r3, 0x4(r4)\nstwu r3, 0x4(r5)\nbdnz .L_1\n")
        self.assertTrue(datagen.copied_to_stack("lbl_80638F24", [asm]))

    def test_definitions_go_after_the_units_statics(self):
        src = "static int s_a;\nvoid f() {\n}\nstatic int s_b;\n"
        a = datagen.Draft("lbl_a", ".bss", 0x10, 4, "verified", text="int lbl_a;")
        self.assertEqual(datagen.apply(src, [a]), src + "\nint lbl_a;\n")


class DestructorRecordTest(unittest.TestCase):
    def test_the_record_passed_to_register_global_object_is_the_compilers(self):
        asm = ("lis r5, lbl_806AE140@ha\naddi r4, r3, __arraydtor$167@l\nli r3, 0x0\n"
               "addi r5, r5, lbl_806AE140@l\nbl __register_global_object\n"
               "li r5, lbl_807920CC@sda21\nbl __register_global_object\nli r5, 0x0\nbl __register_global_object\n")
        self.assertEqual(datagen.destructor_records([asm]), {"lbl_806AE140", "lbl_807920CC"})
        d = datagen.draft(sym("lbl_806AE140", ".bss", 0, 0xC), datagen.items(".skip 0xc"), ".bss", "", ["__sinit_u"],
                          [], [], "local", dtor_record=True)
        self.assertEqual(d.status, "skip")
        self.assertIn("destructor-chain record", d.why)


class PlacementTest(unittest.TestCase):
    def test_an_object_the_source_defines_under_another_name(self):
        # XWidget: the target's lbl_80787E98 is the source's CommandRecord array `s_commands`, and
        # lbl_80792AE8 is vf14's function-local static `init$12`
        target = {("vf14__7XWidgetFv", 0x10): "lbl_80792AE8", ("fn_805E0000", 0x8): "lbl_80787E98",
                  ("fn_805E0000", 0x20): "lbl_80787E98", ("fn_805E0100", 0x4): "lbl_80792AEC"}
        ours = {("vf14__7XWidgetFv", 0x10): "init$12", ("fn_805E0000", 0x8): "s_commands",
                ("fn_805E0000", 0x20): "s_commands", ("fn_805E0100", 0x4): "lbl_80792AEC"}
        got = datagen.same_sites(ours, target, {"init$12", "s_commands"})
        self.assertEqual(got, {"lbl_80792AE8": "init$12", "lbl_80787E98": "s_commands"})

    def test_a_split_vote_is_no_evidence(self):
        target = {("f", 0): "lbl_a", ("f", 4): "lbl_a"}
        ours = {("f", 0): "x", ("f", 4): "y"}
        self.assertEqual(datagen.same_sites(ours, target, {"x", "y"}), {})


class ApplyTest(unittest.TestCase):
    def test_externs_become_definitions_and_the_rest_one_block(self):
        src = "extern int lbl_b;\n// f\nvoid f() {\n}\n"
        a = datagen.Draft("lbl_a", ".bss", 0x10, 4, "verified", text="int lbl_a;")
        b = datagen.Draft("lbl_b", ".bss", 0x14, 4, "verified", text="int lbl_b;", replaces=(0, 18))
        c = datagen.Draft("lbl_c", ".data", 0x20, 4, "verified", text="void (*lbl_c)() = fn_x;",
                          needs=['extern "C" void fn_x();'])
        self.assertEqual(datagen.apply(src, [c, b, a]),
                         'int lbl_b;\n// f\nvoid f() {\n}\n\nextern "C" void fn_x();\nint lbl_a;\nvoid (*lbl_c)() = fn_x;\n')


if __name__ == "__main__":
    unittest.main()


class FourByteGuardTest(unittest.TestCase):
    def test_words_that_read_as_floats_are_not_typed_int(self):
        self.assertTrue(datagen.float_like(0x3F0CCCCD))             # 0.55f
        self.assertTrue(datagen.float_like(0xBF800000))             # -1.0f
        self.assertFalse(datagen.float_like(32))
        self.assertFalse(datagen.float_like(0xFFFFFFFB))            # -5
        self.assertFalse(datagen.float_like(0x7F7FFFFF))            # FLT_MAX: not a plain value
        d = draft(sym("lbl_20", ".sdata", 0, 4, "4byte"), ".4byte 0x3F0CCCCD")
        self.assertEqual(d.status, "todo")
        self.assertIn("read as floats (0.55f)", d.why)
        ok = draft(sym("lbl_21", ".sdata", 0, 4, "4byte"), ".4byte 0x3F0CCCCD", src="extern int lbl_21;\n")
        self.assertEqual(ok.text, "int lbl_21 = 1057803469;")      # the unit's extern is evidence enough

    def test_a_long_4byte_table_without_an_extern_is_a_todo(self):
        words = "\n".join(f".4byte {v}" for v in [6, 12, 0, 53248] * 8)
        d = draft(sym("lbl_22", ".rodata", 0, 128, "4byte"), words)
        self.assertEqual(d.status, "todo")
        self.assertIn("table of 32 words", d.why)
        short = draft(sym("lbl_23", ".rodata", 0, 16, "4byte"), ".4byte 1\n.4byte 2\n.4byte 3\n.4byte 4")
        self.assertEqual(short.text, "const int lbl_23[4] = {1, 2, 3, 4};")
