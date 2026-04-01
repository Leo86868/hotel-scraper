"""POI name utilities — version-aware display name handling.

Versioned names like 'Madonna Inn v2' are used for DB keys and storage paths.
For human-facing output (overlays, hashtags, search queries), use display_name()
to strip the version suffix.
"""

import re

_VERSION_RE = re.compile(r'\s+v\d+$')


def display_name(poi_name: str) -> str:
    """Strip version suffix for display: 'Madonna Inn v2' → 'Madonna Inn'."""
    return _VERSION_RE.sub('', poi_name)
