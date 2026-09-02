class PseudopotentialError(RuntimeError):
    """Base class for user-facing generation failures."""


class ConfigurationError(PseudopotentialError):
    """The request is inconsistent or unsupported."""


class BackendError(PseudopotentialError):
    """A numerical backend failed or produced incomplete artifacts."""


class GhostStateError(PseudopotentialError):
    """No requested Kleinman--Bylander representation passed validation."""
