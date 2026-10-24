import unittest

from tools.dk.platforms.gc_mwcc.demangle import demangle


class DemangleTest(unittest.TestCase):
    def test_free_and_member_functions(self):
        f = demangle("__dl__FPv")
        self.assertEqual((f.quals, f.name, [p.text for p in f.params]), ((), "operator delete", ["void*"]))
        f = demangle("fn_80114DE0__10UnkChannelFi")
        self.assertEqual((f.quals, f.name, [p.text for p in f.params]), (("UnkChannel",), "fn_80114DE0", ["int"]))
        f = demangle("fn_80151EEC__Q24EAGL10SymbolPoolFPCcPv")
        self.assertEqual(f.quals, ("EAGL", "SymbolPool"))
        self.assertEqual([p.text for p in f.params], ["const char*", "void*"])
        self.assertEqual(demangle("vf13__Q22AI9PrototypeFv").params, ())

    def test_types(self):
        f = demangle("__ct__15UnkShaderMethodFPFPv_vPPCcPCiPCc")
        self.assertEqual(f.name, "ctor")
        self.assertEqual([p.text for p in f.params], ["void (*)(void*)", "const char**", "const int*", "const char*"])
        f = demangle("__as__6GUIStrFRC6GUIStr")
        self.assertEqual([p.text for p in f.params], ["const GUIStr&"])
        self.assertEqual(f.params[0].classes, (("GUIStr",),))
        self.assertEqual([p.reg for p in demangle("f__FUlfdi").params], ["gpr", "fpr", "fpr", "gpr"])
        self.assertTrue(demangle("get__5ClassCFv").const)

    def test_refuses_what_it_cannot_reproduce(self):
        self.assertIsNone(demangle("fn_80000000"))
        self.assertIsNone(demangle("push__10Vector<f>Ff"))
        self.assertIsNone(demangle("__sinit_Race_cpp"))
