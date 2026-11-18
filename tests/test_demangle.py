import unittest

from tools.dk.platforms.gc_mwcc.demangle import demangle


class DemangleTest(unittest.TestCase):
    def test_free_and_member_functions(self):
        f = demangle("__dl__FPv")
        self.assertEqual((f.quals, f.name, [p.text for p in f.params]), ((), "operator delete", ["void*"]))
        f = demangle("fn_80514DE0__10DataStreamFi")
        self.assertEqual((f.quals, f.name, [p.text for p in f.params]), (("DataStream",), "fn_80514DE0", ["int"]))
        f = demangle("fn_80551EEC__Q24GFXL10StringPoolFPCcPv")
        self.assertEqual(f.quals, ("GFXL", "StringPool"))
        self.assertEqual([p.text for p in f.params], ["const char*", "void*"])
        self.assertEqual(demangle("vf13__Q22Ns9ArchetypeFv").params, ())

    def test_types(self):
        f = demangle("__ct__15ShaderCallbacksFPFPv_vPPCcPCiPCc")
        self.assertEqual(f.name, "ctor")
        self.assertEqual([p.text for p in f.params], ["void (*)(void*)", "const char**", "const int*", "const char*"])
        f = demangle("__as__6TxtStrFRC6TxtStr")
        self.assertEqual([p.text for p in f.params], ["const TxtStr&"])
        self.assertEqual(f.params[0].classes, (("TxtStr",),))
        self.assertEqual([p.reg for p in demangle("f__FUlfdi").params], ["gpr", "fpr", "fpr", "gpr"])
        self.assertTrue(demangle("get__5ClassCFv").const)

    def test_refuses_what_it_cannot_reproduce(self):
        self.assertIsNone(demangle("fn_80400000"))
        self.assertIsNone(demangle("push__10Vector<f>Ff"))
        self.assertIsNone(demangle("__sinit_Main_cpp"))
