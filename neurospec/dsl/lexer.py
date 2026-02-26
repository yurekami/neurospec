"""Hand-written lexer for the NeuroSpec DSL.

Tokenizes .ns source files into a stream of Token objects.
No external dependencies — pure Python recursive scanning.
"""

from __future__ import annotations

from neurospec.dsl.tokens import KEYWORDS, Token, TokenKind


class LexerError(Exception):
    """Raised when the lexer encounters an invalid character sequence."""

    def __init__(self, message: str, line: int, column: int) -> None:
        self.line = line
        self.column = column
        super().__init__(f"Lexer error at L{line}:{column}: {message}")


class Lexer:
    """Tokenize NeuroSpec DSL source text.

    Usage:
        lexer = Lexer(source_text)
        tokens = lexer.tokenize()
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: list[Token] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tokenize(self) -> list[Token]:
        """Scan the entire source and return all tokens including EOF."""
        while not self._at_end():
            self._skip_whitespace()
            if self._at_end():
                break
            self._scan_token()

        self._tokens.append(Token(TokenKind.EOF, "", self._line, self._col))
        return self._tokens

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_token(self) -> None:
        ch = self._peek()

        # Comments
        if ch == "#":
            self._scan_comment()
            return

        # Newlines
        if ch == "\n":
            self._tokens.append(Token(TokenKind.NEWLINE, "\\n", self._line, self._col))
            self._advance()
            return

        # Strings
        if ch == '"':
            self._scan_string()
            return

        # Numbers (including negative)
        if ch.isdigit() or (ch == "-" and self._peek_next().isdigit()):
            self._scan_number()
            return

        # Identifiers / keywords
        if ch.isalpha() or ch == "_":
            self._scan_identifier()
            return

        # Single-character punctuation
        punct_map = {
            "{": TokenKind.LBRACE,
            "}": TokenKind.RBRACE,
            "(": TokenKind.LPAREN,
            ")": TokenKind.RPAREN,
            "[": TokenKind.LBRACKET,
            "]": TokenKind.RBRACKET,
            ":": TokenKind.COLON,
            ",": TokenKind.COMMA,
            ".": TokenKind.DOT,
            "=": TokenKind.EQUALS,
            "<": TokenKind.LT,
            ">": TokenKind.GT,
        }

        if ch in punct_map:
            self._tokens.append(Token(punct_map[ch], ch, self._line, self._col))
            self._advance()
            return

        raise LexerError(f"Unexpected character: {ch!r}", self._line, self._col)

    def _scan_comment(self) -> None:
        """Consume a # comment until end of line."""
        start_col = self._col
        self._advance()  # skip #
        text = ""
        while not self._at_end() and self._peek() != "\n":
            text += self._peek()
            self._advance()
        # Comments are skipped (not added to token stream)

    def _scan_string(self) -> None:
        """Scan a double-quoted string literal."""
        start_line = self._line
        start_col = self._col
        self._advance()  # skip opening quote

        text = ""
        while not self._at_end():
            ch = self._peek()
            if ch == '"':
                self._advance()  # skip closing quote
                self._tokens.append(Token(TokenKind.STRING, text, start_line, start_col))
                return
            if ch == "\\":
                self._advance()
                escaped = self._peek()
                escape_map = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}
                text += escape_map.get(escaped, escaped)
                self._advance()
            else:
                if ch == "\n":
                    raise LexerError(
                        "Unterminated string (newline before closing quote)", start_line, start_col
                    )
                text += ch
                self._advance()

        raise LexerError("Unterminated string (hit EOF)", start_line, start_col)

    def _scan_number(self) -> None:
        """Scan a numeric literal (int or float, with optional exponent)."""
        start_col = self._col
        text = ""

        if self._peek() == "-":
            text += "-"
            self._advance()

        while not self._at_end() and self._peek().isdigit():
            text += self._peek()
            self._advance()

        if not self._at_end() and self._peek() == ".":
            text += "."
            self._advance()
            while not self._at_end() and self._peek().isdigit():
                text += self._peek()
                self._advance()

        # Scientific notation
        if not self._at_end() and self._peek() in ("e", "E"):
            text += self._peek()
            self._advance()
            if not self._at_end() and self._peek() in ("+", "-"):
                text += self._peek()
                self._advance()
            while not self._at_end() and self._peek().isdigit():
                text += self._peek()
                self._advance()

        self._tokens.append(Token(TokenKind.NUMBER, text, self._line, start_col))

    def _scan_identifier(self) -> None:
        """Scan an identifier or keyword."""
        start_col = self._col
        text = ""

        while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
            text += self._peek()
            self._advance()

        kind = KEYWORDS.get(text, TokenKind.IDENTIFIER)
        self._tokens.append(Token(kind, text, self._line, start_col))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _at_end(self) -> bool:
        return self._pos >= len(self._source)

    def _peek(self) -> str:
        if self._at_end():
            return "\0"
        return self._source[self._pos]

    def _peek_next(self) -> str:
        if self._pos + 1 >= len(self._source):
            return "\0"
        return self._source[self._pos + 1]

    def _advance(self) -> str:
        ch = self._source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _skip_whitespace(self) -> None:
        """Skip spaces, tabs, and carriage returns (but not newlines)."""
        while not self._at_end() and self._peek() in (" ", "\t", "\r"):
            self._advance()
