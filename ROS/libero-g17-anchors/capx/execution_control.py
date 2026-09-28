"""Control-flow signals shared by generated-code APIs and the executor."""

from __future__ import annotations


class CodeBlockBreakpoint(BaseException):
    """End the current generated block successfully and request observation.

    This derives from :class:`BaseException` so generated code that catches a
    broad ``Exception`` cannot accidentally consume the transaction boundary.
    The code executor handles it explicitly and reports sandbox success.
    """

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def breakpoint_code_block(reason: str) -> None:
    """Raise the normal executor signal for an observation/regeneration turn."""

    raise CodeBlockBreakpoint(reason)
