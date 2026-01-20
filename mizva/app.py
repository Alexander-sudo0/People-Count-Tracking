"""Compatibility shim: re-export the FaceAnalysis API from the original package.

This allows code to `from mizva.app import FaceAnalysis` while keeping the original
`insightface` implementation files untouched. Later we can replace the implementation
with a native MizVa implementation.
"""
try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:
    # Provide a clear ImportError if the original package is not present.
    raise ImportError('The original insightface package is required for the mizva shim to work.\n'
                      'Install the project dependencies or convert the implementation to MizVa.')

__all__ = ['FaceAnalysis']
