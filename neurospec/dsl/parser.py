"""Hand-written recursive descent parser for the NeuroSpec DSL.

Consumes a list of Token objects from the lexer and produces a SpecFile AST.
No external parsing libraries used.
"""

from __future__ import annotations

from typing import Any

from neurospec.dsl.ast_nodes import (
    AlertIfRule,
    AmplifyRule,
    CompileToWeightsRule,
    ComposeDecl,
    ImportDecl,
    ModelDecl,
    MonitorRule,
    OverrideDecl,
    RequireRule,
    SAEDecl,
    SpecDecl,
    SpecFile,
    SuppressRule,
)
from neurospec.dsl.tokens import Token, TokenKind


class ParseError(Exception):
    """Raised when the parser encounters an unexpected token."""

    def __init__(self, message: str, token: Token) -> None:
        self.token = token
        super().__init__(f"Parse error at L{token.line}:{token.column}: {message}")


class Parser:
    """Parse a NeuroSpec token stream into a SpecFile AST.

    Usage:
        parser = Parser(tokens)
        spec_file = parser.parse()
    """

    def __init__(self, tokens: list[Token]) -> None:
        # Filter out newlines — they're syntactically insignificant in our grammar
        self._tokens = [t for t in tokens if t.kind != TokenKind.NEWLINE]
        self._pos = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self) -> SpecFile:
        """Parse the full token stream into a SpecFile."""
        spec_file = SpecFile()

        while not self._at_end():
            if self._check(TokenKind.MODEL):
                spec_file.model = self._parse_model()
            elif self._check(TokenKind.SAE):
                spec_file.sae = self._parse_sae()
            elif self._check(TokenKind.IMPORT):
                spec_file.imports.append(self._parse_import())
            elif self._check(TokenKind.SPEC):
                result = self._parse_spec_or_compose()
                if isinstance(result, SpecDecl):
                    spec_file.specs.append(result)
                else:
                    spec_file.composes.append(result)
            elif self._check(TokenKind.EOF):
                break
            else:
                raise ParseError(
                    f"Unexpected token: {self._current().kind.name}",
                    self._current(),
                )

        return spec_file

    # ------------------------------------------------------------------
    # Top-level declarations
    # ------------------------------------------------------------------

    def _parse_model(self) -> ModelDecl:
        token = self._consume(TokenKind.MODEL, "Expected 'model'")
        model_id = self._consume(TokenKind.STRING, "Expected model ID string").value
        return ModelDecl(model_id=model_id, line=token.line)

    def _parse_sae(self) -> SAEDecl:
        token = self._consume(TokenKind.SAE, "Expected 'sae'")
        sae_id = self._consume(TokenKind.STRING, "Expected SAE ID string").value
        return SAEDecl(sae_id=sae_id, line=token.line)

    def _parse_import(self) -> ImportDecl:
        token = self._consume(TokenKind.IMPORT, "Expected 'import'")
        path = self._consume(TokenKind.STRING, "Expected import path string").value
        self._consume(TokenKind.AS, "Expected 'as' after import path")
        alias = self._consume(TokenKind.IDENTIFIER, "Expected alias identifier").value
        return ImportDecl(path=path, alias=alias, line=token.line)

    def _parse_spec_or_compose(self) -> SpecDecl | ComposeDecl:
        """Parse either `spec name { ... }` or `spec name = compose(...) { ... }`."""
        token = self._consume(TokenKind.SPEC, "Expected 'spec'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected spec name").value

        # Check for compose: `spec name = compose(...) { ... }`
        if self._check(TokenKind.EQUALS):
            self._advance()
            self._consume(TokenKind.COMPOSE, "Expected 'compose' after '='")
            return self._parse_compose_body(name, token.line)

        # Regular spec block
        self._consume(TokenKind.LBRACE, "Expected '{' after spec name")
        rules = self._parse_rules()
        self._consume(TokenKind.RBRACE, "Expected '}' to close spec block")
        return SpecDecl(name=name, rules=rules, line=token.line)

    def _parse_compose_body(self, name: str, line: int) -> ComposeDecl:
        """Parse `(source1, source2) { override ... }`."""
        self._consume(TokenKind.LPAREN, "Expected '(' after 'compose'")
        sources: list[str] = []
        sources.append(self._consume(TokenKind.IDENTIFIER, "Expected source spec name").value)
        while self._check(TokenKind.COMMA):
            self._advance()
            sources.append(self._consume(TokenKind.IDENTIFIER, "Expected source spec name").value)
        self._consume(TokenKind.RPAREN, "Expected ')' after compose sources")

        overrides: list[OverrideDecl] = []
        if self._check(TokenKind.LBRACE):
            self._advance()
            while not self._check(TokenKind.RBRACE) and not self._at_end():
                if self._check(TokenKind.OVERRIDE):
                    overrides.append(self._parse_override())
                else:
                    raise ParseError(
                        f"Expected 'override' in compose block, got {self._current().kind.name}",
                        self._current(),
                    )
            self._consume(TokenKind.RBRACE, "Expected '}' to close compose block")

        return ComposeDecl(name=name, sources=sources, overrides=overrides, line=line)

    def _parse_override(self) -> OverrideDecl:
        token = self._consume(TokenKind.OVERRIDE, "Expected 'override'")
        # Parse dotted path: a.b.c
        parts = [self._consume(TokenKind.IDENTIFIER, "Expected override target").value]
        while self._check(TokenKind.DOT):
            self._advance()
            parts.append(self._consume(TokenKind.IDENTIFIER, "Expected identifier after '.'").value)
        target = ".".join(parts)

        self._consume(TokenKind.COLON, "Expected ':' after override target")
        value = self._parse_value()
        return OverrideDecl(target=target, value=value, line=token.line)

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _parse_rules(self) -> list[Any]:
        """Parse rule declarations inside a spec block."""
        rules: list[Any] = []
        while not self._check(TokenKind.RBRACE) and not self._at_end():
            if self._check(TokenKind.SUPPRESS):
                rules.append(self._parse_suppress())
            elif self._check(TokenKind.AMPLIFY):
                rules.append(self._parse_amplify())
            elif self._check(TokenKind.REQUIRE):
                rules.append(self._parse_require())
            elif self._check(TokenKind.MONITOR):
                rules.append(self._parse_monitor())
            elif self._check(TokenKind.ALERT_IF):
                rules.append(self._parse_alert_if())
            elif self._check(TokenKind.COMPILE_TO_WEIGHTS):
                rules.append(self._parse_compile_to_weights())
            else:
                raise ParseError(
                    f"Expected rule keyword, got {self._current().kind.name}",
                    self._current(),
                )
        return rules

    def _parse_suppress(self) -> SuppressRule:
        token = self._consume(TokenKind.SUPPRESS, "Expected 'suppress'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected rule name").value
        self._consume(TokenKind.LBRACE, "Expected '{'")
        props = self._parse_properties()
        self._consume(TokenKind.RBRACE, "Expected '}'")

        action_str = str(props.get("action", "steer_away"))
        action_name, action_params = self._parse_action_call(action_str)

        return SuppressRule(
            name=name,
            features=props.get("features", []),
            action=action_name,
            action_params=action_params,
            strength=float(props.get("strength", 0.5)),
            line=token.line,
        )

    def _parse_amplify(self) -> AmplifyRule:
        token = self._consume(TokenKind.AMPLIFY, "Expected 'amplify'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected rule name").value
        self._consume(TokenKind.LBRACE, "Expected '{'")
        props = self._parse_properties()
        self._consume(TokenKind.RBRACE, "Expected '}'")

        return AmplifyRule(
            name=name,
            features=props.get("features", []),
            strength=float(props.get("strength", 0.5)),
            line=token.line,
        )

    def _parse_require(self) -> RequireRule:
        token = self._consume(TokenKind.REQUIRE, "Expected 'require'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected rule name").value
        self._consume(TokenKind.LBRACE, "Expected '{'")
        props = self._parse_properties()
        self._consume(TokenKind.RBRACE, "Expected '}'")

        return RequireRule(
            name=name,
            when=str(props.get("when", "")),
            amplify_features=props.get("amplify", []),
            strength=float(props.get("strength", 0.5)),
            line=token.line,
        )

    def _parse_monitor(self) -> MonitorRule:
        token = self._consume(TokenKind.MONITOR, "Expected 'monitor'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected rule name").value
        self._consume(TokenKind.LBRACE, "Expected '{'")
        props = self._parse_properties()
        self._consume(TokenKind.RBRACE, "Expected '}'")

        action_str = str(props.get("action", "log"))
        action_name, action_params = self._parse_action_call(action_str)

        return MonitorRule(
            name=name,
            features=props.get("features", []),
            threshold=float(props.get("threshold", 0.5)),
            action=action_name,
            action_params=action_params,
            line=token.line,
        )

    def _parse_alert_if(self) -> AlertIfRule:
        token = self._consume(TokenKind.ALERT_IF, "Expected 'alert_if'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected rule name").value
        self._consume(TokenKind.LBRACE, "Expected '{'")
        props = self._parse_properties()
        self._consume(TokenKind.RBRACE, "Expected '}'")

        return AlertIfRule(
            name=name,
            features=props.get("features", []),
            threshold=float(props.get("threshold", 0.5)),
            severity=str(props.get("severity", "warning")),
            message=str(props.get("message", "")),
            line=token.line,
        )

    def _parse_compile_to_weights(self) -> CompileToWeightsRule:
        token = self._consume(TokenKind.COMPILE_TO_WEIGHTS, "Expected 'compile_to_weights'")
        name = self._consume(TokenKind.IDENTIFIER, "Expected rule name").value
        self._consume(TokenKind.LBRACE, "Expected '{'")
        props = self._parse_properties()
        self._consume(TokenKind.RBRACE, "Expected '}'")

        return CompileToWeightsRule(
            name=name,
            method=str(props.get("method", "rlfr")),
            probe_source=str(props.get("probe_source", "catalog")),
            training_budget=int(props.get("training_budget", 500)),
            verify_no_regression=props.get("verify_no_regression", []),
            line=token.line,
        )

    # ------------------------------------------------------------------
    # Property blocks
    # ------------------------------------------------------------------

    def _parse_properties(self) -> dict[str, Any]:
        """Parse key: value pairs inside a { ... } block."""
        props: dict[str, Any] = {}

        while not self._check(TokenKind.RBRACE) and not self._at_end():
            key_token = self._current()

            # The key can be a keyword token (features, strength, etc.) or an identifier
            if key_token.kind in (
                TokenKind.FEATURES,
                TokenKind.STRENGTH,
                TokenKind.THRESHOLD,
                TokenKind.METHOD,
                TokenKind.PROBE_SOURCE,
                TokenKind.TRAINING_BUDGET,
                TokenKind.VERIFY_NO_REGRESSION,
                TokenKind.ACTION,
                TokenKind.WHEN,
                TokenKind.IDENTIFIER,
                TokenKind.AMPLIFY,
            ):
                key = key_token.value
                self._advance()
            else:
                raise ParseError(
                    f"Expected property name, got {key_token.kind.name}",
                    key_token,
                )

            self._consume(TokenKind.COLON, f"Expected ':' after property '{key}'")
            value = self._parse_value()
            props[key] = value

        return props

    def _parse_value(self) -> Any:
        """Parse a property value: string, number, list, or action call."""
        token = self._current()

        if token.kind == TokenKind.STRING:
            self._advance()
            return token.value

        if token.kind == TokenKind.NUMBER:
            self._advance()
            # Return int if no decimal point, else float
            if "." in token.value or "e" in token.value.lower():
                return float(token.value)
            return int(token.value)

        if token.kind == TokenKind.LBRACKET:
            return self._parse_list()

        # Action call: identifier(params) e.g., steer_away(strength=0.8)
        if token.kind == TokenKind.IDENTIFIER:
            self._advance()
            if self._check(TokenKind.LPAREN):
                # It's a function call like pause_and_retry(max_attempts=5)
                return self._parse_call_value(token.value)

            # Check for dotted expression: a.b < 0.5
            if self._check(TokenKind.DOT):
                text = token.value
                while self._check(TokenKind.DOT):
                    self._advance()
                    next_part = self._consume(
                        TokenKind.IDENTIFIER, "Expected identifier after '.'"
                    ).value
                    text += f".{next_part}"
                # Check for comparison operator
                if self._check(TokenKind.LT) or self._check(TokenKind.GT):
                    op = self._current().value
                    self._advance()
                    right = self._parse_value()
                    return f"{text} {op} {right}"
                return text

            return token.value

        # Keyword used as value (e.g., `probe_source: catalog`)
        if token.kind in (
            TokenKind.METHOD,
            TokenKind.PROBE_SOURCE,
            TokenKind.MODEL,
            TokenKind.SAE,
            TokenKind.MONITOR,
            TokenKind.AMPLIFY,
        ):
            self._advance()
            return token.value

        raise ParseError(f"Expected value, got {token.kind.name}", token)

    def _parse_list(self) -> list[Any]:
        """Parse a bracketed list: [value, value, ...]."""
        self._consume(TokenKind.LBRACKET, "Expected '['")
        items: list[Any] = []

        if not self._check(TokenKind.RBRACKET):
            items.append(self._parse_value())
            while self._check(TokenKind.COMMA):
                self._advance()
                if self._check(TokenKind.RBRACKET):
                    break  # trailing comma
                items.append(self._parse_value())

        self._consume(TokenKind.RBRACKET, "Expected ']'")
        return items

    def _parse_call_value(self, name: str) -> str:
        """Parse a function-call-style value like `steer_away(strength=0.8)`.

        Returns the full call as a string for later interpretation by the compiler.
        """
        self._consume(TokenKind.LPAREN, "Expected '('")
        parts = [name, "("]
        depth = 1
        while depth > 0 and not self._at_end():
            token = self._current()
            if token.kind == TokenKind.LPAREN:
                depth += 1
            elif token.kind == TokenKind.RPAREN:
                depth -= 1
                if depth == 0:
                    self._advance()
                    break
            parts.append(token.value)
            self._advance()
        return "".join(parts) + ")"

    # ------------------------------------------------------------------
    # Action parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action_call(action_str: str) -> tuple[str, dict[str, Any]]:
        """Extract action name and params from a call string like `steer_away(strength=0.8)`.

        Returns:
            Tuple of (action_name, params_dict).
        """
        if "(" not in action_str:
            return action_str, {}

        name = action_str[: action_str.index("(")]
        params_str = action_str[action_str.index("(") + 1 : action_str.rindex(")")]
        params: dict[str, Any] = {}

        if params_str.strip():
            for part in params_str.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Try to parse as number
                    try:
                        params[k] = int(v)
                    except ValueError:
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v

        return name, params

    # ------------------------------------------------------------------
    # Token stream helpers
    # ------------------------------------------------------------------

    def _current(self) -> Token:
        if self._pos >= len(self._tokens):
            return Token(TokenKind.EOF, "", 0, 0)
        return self._tokens[self._pos]

    def _at_end(self) -> bool:
        return self._current().kind == TokenKind.EOF

    def _check(self, kind: TokenKind) -> bool:
        return self._current().kind == kind

    def _advance(self) -> Token:
        token = self._current()
        if not self._at_end():
            self._pos += 1
        return token

    def _consume(self, kind: TokenKind, message: str) -> Token:
        if self._check(kind):
            return self._advance()
        raise ParseError(
            f"{message} (got {self._current().kind.name}: {self._current().value!r})",
            self._current(),
        )
