"""Offline four-part singing and rehearsal-track production."""
import os

__version__ = '0.1.0'


def default_threads():
    """Respect CPU affinity without oversubscribing small inference chunks."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(8, available))
