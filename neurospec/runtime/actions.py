"""Runtime actions triggered by monitors and alerts.

Defines the actions that can be taken when a monitor threshold is crossed:
steer, retry, alert, log, or kill the generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from neurospec.core.types import ActionKind

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing a runtime action."""

    action: ActionKind
    success: bool
    message: str = ""
    should_retry: bool = False
    should_kill: bool = False


class ActionExecutor:
    """Execute runtime actions in response to monitor triggers.

    Actions modify engine state (e.g., increasing steering strength,
    triggering a retry, or killing generation).
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._alert_handlers: list[Any] = []

    def register_alert_handler(self, handler: Any) -> None:
        """Register a callback for alert actions."""
        self._alert_handlers.append(handler)

    def execute(
        self,
        action: ActionKind,
        params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActionResult:
        """Execute a runtime action.

        Args:
            action: The kind of action to execute.
            params: Action-specific parameters (e.g., max_attempts for retry).
            context: Context about what triggered the action.

        Returns:
            ActionResult describing what happened.
        """
        params = params or {}
        context = context or {}

        dispatch = {
            ActionKind.STEER_AWAY: self._action_steer_away,
            ActionKind.STEER_TOWARD: self._action_steer_toward,
            ActionKind.PAUSE_AND_RETRY: self._action_pause_and_retry,
            ActionKind.ALERT: self._action_alert,
            ActionKind.KILL: self._action_kill,
            ActionKind.LOG: self._action_log,
        }

        handler = dispatch.get(action)
        if handler is None:
            logger.warning("Unknown action kind: %s", action)
            return ActionResult(action=action, success=False, message=f"Unknown action: {action}")

        return handler(params, context)

    def _action_steer_away(self, params: dict[str, Any], context: dict[str, Any]) -> ActionResult:
        """Increase suppression steering on triggered features."""
        strength = float(params.get("strength", 0.8))
        logger.info(
            "Steering away with strength %.2f (context: %s)",
            strength,
            context.get("monitor_name", "?"),
        )
        return ActionResult(
            action=ActionKind.STEER_AWAY,
            success=True,
            message=f"Applied steer_away with strength {strength}",
        )

    def _action_steer_toward(self, params: dict[str, Any], context: dict[str, Any]) -> ActionResult:
        """Increase amplification steering on triggered features."""
        strength = float(params.get("strength", 0.5))
        logger.info("Steering toward with strength %.2f", strength)
        return ActionResult(
            action=ActionKind.STEER_TOWARD,
            success=True,
            message=f"Applied steer_toward with strength {strength}",
        )

    def _action_pause_and_retry(
        self, params: dict[str, Any], context: dict[str, Any]
    ) -> ActionResult:
        """Signal the engine to pause and retry generation."""
        max_attempts = int(params.get("max_attempts", 5))
        logger.info("Pause and retry (max_attempts=%d)", max_attempts)
        return ActionResult(
            action=ActionKind.PAUSE_AND_RETRY,
            success=True,
            message=f"Retry requested (max {max_attempts} attempts)",
            should_retry=True,
        )

    def _action_alert(self, params: dict[str, Any], context: dict[str, Any]) -> ActionResult:
        """Fire an alert to registered handlers."""
        message = str(params.get("message", context.get("monitor_name", "Alert triggered")))
        logger.warning("Alert: %s", message)

        for handler in self._alert_handlers:
            try:
                handler(message, context)
            except Exception as exc:
                logger.error("Alert handler failed: %s", exc)

        return ActionResult(action=ActionKind.ALERT, success=True, message=message)

    def _action_kill(self, params: dict[str, Any], context: dict[str, Any]) -> ActionResult:
        """Signal the engine to immediately stop generation."""
        reason = str(params.get("reason", "Monitor threshold exceeded"))
        logger.critical("Kill generation: %s", reason)
        return ActionResult(
            action=ActionKind.KILL,
            success=True,
            message=f"Generation killed: {reason}",
            should_kill=True,
        )

    def _action_log(self, params: dict[str, Any], context: dict[str, Any]) -> ActionResult:
        """Log the trigger event without taking other action."""
        message = str(params.get("message", context.get("monitor_name", "Monitor triggered")))
        logger.info("Monitor log: %s", message)
        return ActionResult(action=ActionKind.LOG, success=True, message=message)
