from __future__ import annotations

from heapq import heappop, heappush
from itertools import product
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .grammar import NonTerminal, Rule, Symbol
from .lexer import PatternRE, PatternStr, TerminalDef


_INFINITY = 10 ** 12


class GrammarGenerator:
    """Generate example sentences from a compiled Lark grammar.

    The generator performs a best-first expansion over the grammar rules,
    preferring states with a lower estimated output length.
    """

    def __init__(self, rules: Sequence[Rule], terminals: Sequence[TerminalDef]) -> None:
        self._rules_by_origin: Dict[str, List[Rule]] = {}
        for rule in rules:
            self._rules_by_origin.setdefault(rule.origin.name, []).append(rule)

        self._terminal_samples: Dict[str, str] = {
            t.name: self._sample_terminal(t) for t in terminals
        }

        # Lower-bound estimate used to prioritize expansion
        self._symbol_min_len: Dict[Tuple[bool, str], int] = {}
        for name, sample in self._terminal_samples.items():
            self._symbol_min_len[(True, name)] = len(sample)
        for nt in self._rules_by_origin:
            self._symbol_min_len[(False, nt)] = _INFINITY

        changed = True
        while changed:
            changed = False
            for nt, rules in self._rules_by_origin.items():
                best = self._symbol_min_len[(False, nt)]
                for rule in rules:
                    total = 0
                    for sym in rule.expansion:
                        total += self._symbol_min_len.get((sym.is_term, sym.name), _INFINITY)
                        if total >= best:
                            break
                    if total < best:
                        best = total
                if best < self._symbol_min_len[(False, nt)]:
                    self._symbol_min_len[(False, nt)] = best
                    changed = True

    @classmethod
    def from_lark(cls, lark_inst) -> "GrammarGenerator":
        return cls(lark_inst.rules, lark_inst.terminals)

    def iter_generate(self, start: str, max_depth: int = 20, max_results: Optional[int] = None) -> Iterator[str]:
        agenda: List[Tuple[int, int, int, Tuple[Symbol, ...]]] = []
        counter = 0
        start_form = (NonTerminal(start),)
        heappush(agenda, (self._estimate_form_len(start_form), 0, counter, start_form))

        seen = set()
        yielded = set()

        while agenda:
            _est_len, depth, _id, form = heappop(agenda)
            if form in seen:
                continue
            seen.add(form)

            if all(sym.is_term for sym in form):
                out = ''.join(self._terminal_samples[sym.name] for sym in form)
                if out not in yielded:
                    yielded.add(out)
                    yield out
                    if max_results is not None and len(yielded) >= max_results:
                        return
                continue

            if depth >= max_depth:
                continue

            idx = next(i for i, sym in enumerate(form) if not sym.is_term)
            nt = form[idx]
            assert isinstance(nt, NonTerminal)

            for rule in self._rules_by_origin.get(nt.name, []):
                expanded = form[:idx] + tuple(rule.expansion) + form[idx + 1 :]
                counter += 1
                heappush(agenda, (self._estimate_form_len(expanded), depth + 1, counter, expanded))

    def _estimate_form_len(self, form: Iterable[Symbol]) -> int:
        return sum(self._symbol_min_len.get((sym.is_term, sym.name), _INFINITY) for sym in form)

    def _sample_terminal(self, terminal: TerminalDef) -> str:
        pattern = terminal.pattern
        if isinstance(pattern, PatternStr):
            return pattern.value

        assert isinstance(pattern, PatternRE)
        source = pattern.to_regexp()
        cre = re.compile(source)

        candidates = self._regex_heuristics(pattern.value)
        for candidate in candidates:
            if cre.fullmatch(candidate):
                return candidate

        min_width = max(pattern.min_width, 0)
        alphabet = "abcxyzABCXYZ0123_ -"
        max_try = min(min_width + 4, 8)
        for n in range(min_width, max_try + 1):
            for chars in product(alphabet, repeat=n):
                candidate = ''.join(chars)
                if cre.fullmatch(candidate):
                    return candidate

        raise ValueError("Unable to generate a sample value for terminal %r" % terminal.name)

    def _regex_heuristics(self, raw_pattern: str) -> List[str]:
        base = [
            "",
            "a",
            "0",
            "_",
            " ",
            "aa",
            "00",
            "a0",
        ]
        if r"\d" in raw_pattern:
            base.extend(["0", "1", "00", "01"])
        if r"\w" in raw_pattern:
            base.extend(["a", "A", "a0", "_"])
        if "[" in raw_pattern:
            m = re.search(r"\[([^\]]+)\]", raw_pattern)
            if m and m.group(1):
                chunk = m.group(1)
                base.append(chunk[0])
        return base
