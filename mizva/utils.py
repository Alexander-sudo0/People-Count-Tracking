"""Utility shim for mizva — proxies to insightface utils when available.
"""
try:
    from insightface.utils import face_align  # type: ignore
except Exception:
    def face_align(img, landmark, output_size=112):
        raise NotImplementedError('face_align shim is not implemented. Install insightface or implement MizVa\'s utilities.')

__all__ = ['face_align']
