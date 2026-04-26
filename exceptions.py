
class RetrievalError(Exception):
    """Base exception for retrieval/indexing errors."""


class IndexNotBuiltError(RetrievalError):
    """Raised when an operation requires a built or loaded index."""


class InvalidVectorShapeError(RetrievalError):
    """Raised when vectors or query vectors have an invalid shape."""


class InvalidVectorTypeError(RetrievalError):
    """Raised when vectors are not provided as a NumPy array."""


class InvalidVectorValueError(RetrievalError):
    """Raised when vectors contain NaN, inf, or 

unsupported values."""


class ZeroVectorError(RetrievalError):
    """Raised when cosine normalization is attempted on a zero vector."""


class MetadataMismatchError(RetrievalError):
    """Raised when metadata does not align with vectors."""


class IndexPersistenceError(RetrievalError):
    """Raised when saving or loading index artifacts fails."""


class ConfigurationError(RetrievalError):
    """Raised when saved config is missing or inconsistent."""
