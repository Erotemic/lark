import unittest

from lark import Lark


class TestGenerator(unittest.TestCase):
    def test_iter_generate_prefers_short_derivations(self):
        p = Lark(r'''
            start: atom | atom atom
            atom: "a" | "b"
        ''', parser='lalr')

        results = list(p.iter_generate(max_results=6))

        self.assertEqual(results[:2], ['a', 'b'])
        self.assertIn('aa', results)
        self.assertIn('ab', results)

    def test_iter_generate_handles_simple_regex_terminal(self):
        p = Lark(r'''
            start: ID
            ID: /[a-z][a-z0-9_]*/
        ''', parser='lalr')

        value, = list(p.iter_generate(max_results=1))
        self.assertTrue(value)
        self.assertTrue(value[0].islower())


if __name__ == '__main__':
    unittest.main()
