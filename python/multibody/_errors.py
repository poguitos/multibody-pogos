"""The exception and warning classes of the multibody package (D52).

Every error and warning of the engine carries a code, ``MBD-`` with a letter
for the area and three digits, at the start of its message; the catalogue of
codes, with what each means and what to do, is docs/help/messages.md. The
classes keep the whole message and read the code from it.
"""

import re

_CODE = re.compile(r"MBD-[A-Z][0-9]{3}")


def _code_of(message):
    match = _CODE.match(message)
    return match.group(0) if match else None


class MbdError(RuntimeError):
    """An error of the engine: an ``mbd::MbdError`` thrown in C++."""

    @property
    def code(self):
        """The message's code, such as ``"MBD-K051"``, or None."""
        return _code_of(str(self))


class MbdWarning(UserWarning):
    """A warning of the engine, sent through its diagnostic sink."""

    @property
    def code(self):
        """The message's code, such as ``"MBD-K050"``, or None."""
        return _code_of(str(self))
