"""Token types for the NeuroSpec DSL lexer.

Defines all token kinds and the Token dataclass used by the lexer and parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    """All token types recognized by the NeuroSpec lexer."""

    # Literals
    STRING = auto()  # "quoted string"
    NUMBER = auto()  # 0.8, 42, 1e-5
    IDENTIFIER = auto()  # unquoted name

    # Keywords
    MODEL = auto()  # model
    SAE = auto()  # sae
    SPEC = auto()  # spec
    SUPPRESS = auto()  # suppress
    AMPLIFY = auto()  # amplify
    REQUIRE = auto()  # require
    MONITOR = auto()  # monitor
    ALERT_IF = auto()  # alert_if
    COMPILE_TO_WEIGHTS = auto()  # compile_to_weights
    IMPORT = auto()  # import
    AS = auto()  # as
    COMPOSE = auto()  # compose
    OVERRIDE = auto()  # override
    WHEN = auto()  # when
    ACTION = auto()  # action
    FEATURES = auto()  # features
    STRENGTH = auto()  # strength
    THRESHOLD = auto()  # threshold
    METHOD = auto()  # method
    PROBE_SOURCE = auto()  # probe_source
    TRAINING_BUDGET = auto()  # training_budget
    VERIFY_NO_REGRESSION = auto()  # verify_no_regression

    # Punctuation
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COLON = auto()  # :
    COMMA = auto()  # ,
    DOT = auto()  # .
    EQUALS = auto()  # =
    LT = auto()  # <
    GT = auto()  # >

    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()  # # comment


# Map keyword strings to token kinds
KEYWORDS: dict[str, TokenKind] = {
    "model": TokenKind.MODEL,
    "sae": TokenKind.SAE,
    "spec": TokenKind.SPEC,
    "suppress": TokenKind.SUPPRESS,
    "amplify": TokenKind.AMPLIFY,
    "require": TokenKind.REQUIRE,
    "monitor": TokenKind.MONITOR,
    "alert_if": TokenKind.ALERT_IF,
    "compile_to_weights": TokenKind.COMPILE_TO_WEIGHTS,
    "import": TokenKind.IMPORT,
    "as": TokenKind.AS,
    "compose": TokenKind.COMPOSE,
    "override": TokenKind.OVERRIDE,
    "when": TokenKind.WHEN,
    "action": TokenKind.ACTION,
    "features": TokenKind.FEATURES,
    "strength": TokenKind.STRENGTH,
    "threshold": TokenKind.THRESHOLD,
    "method": TokenKind.METHOD,
    "probe_source": TokenKind.PROBE_SOURCE,
    "training_budget": TokenKind.TRAINING_BUDGET,
    "verify_no_regression": TokenKind.VERIFY_NO_REGRESSION,
}


@dataclass(frozen=True)
class Token:
    """A single token produced by the lexer."""

    kind: TokenKind
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.value!r}, L{self.line}:{self.column})"
