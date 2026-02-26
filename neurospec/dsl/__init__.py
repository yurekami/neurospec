"""NeuroSpec DSL — tokenizer, parser, and validator for .ns spec files.

Usage:
    from neurospec.dsl import Lexer, Parser, validate_spec

    tokens = Lexer(source).tokenize()
    spec_file = Parser(tokens).parse()
    errors = validate_spec(spec_file)
"""

from neurospec.dsl.lexer import Lexer, LexerError
from neurospec.dsl.parser import ParseError, Parser
from neurospec.dsl.validator import validate_spec

__all__ = [
    "Lexer",
    "LexerError",
    "ParseError",
    "Parser",
    "validate_spec",
]
