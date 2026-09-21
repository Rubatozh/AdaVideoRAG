"""Seams the multimodal-RAG benchmark installs at runtime.

Not upstream. Every hook defaults to identity: with nothing installed, the
upstream execution path runs unchanged. Inserted call sites are marked with
`# BENCH SEAM BEGIN/END`.
"""

_HOOKS = {}
_FIRED = {}


def install(name, fn):
    """Install a hook. `fn(value, **kw) -> value` must be shape-preserving."""
    _HOOKS[name] = fn


def clear():
    _HOOKS.clear()
    _FIRED.clear()


def installed():
    return sorted(_HOOKS)


def fired():
    """{hook: n_calls}."""
    return dict(_FIRED)


def _apply(name, value, **kw):
    fn = _HOOKS.get(name)
    if fn is None:
        return value                      # <- the upstream path, unchanged
    _FIRED[name] = _FIRED.get(name, 0) + 1
    return fn(value, **kw)


def _optional(name, **kw):
    """For hooks that produce something or decline. -> value or None."""
    fn = _HOOKS.get(name)
    if fn is None:
        return None                       # <- the upstream path, unchanged
    _FIRED[name] = _FIRED.get(name, 0) + 1
    return fn(**kw)


# --------------------------------------------------------------- the seams
def frame_times(times, **kw):
    """INPUT POOL, moments. Substitutes the pool's timestamps for the
    per-segment sample `deal_video` computed."""
    return _apply("frame_times", times, **kw)


def frame_pixels(image, **kw):
    """INPUT POOL, pixels (single frame). Returns the pool's frame for this
    moment instead of the frame decoded from the source clip."""
    return _apply("frame_pixels", image, **kw)


def frames_for(frames, **kw):
    """INPUT POOL, pixels (a segment's caption frames). Substitutes the
    pool's frames for the ones `encode_video` decoded."""
    return _apply("frames_for", frames, **kw)


def segment_clip(**kw):
    """INPUT POOL, ImageBind. Writes the segment's video file from pool
    frames.

    Returns True when it wrote the file. None/False means not handled, and
    upstream's own `write_videofile` runs exactly as before.
    """
    return _optional("segment_clip", **kw)


def retrieved(segments, **kw):
    """MODULES, retrieve. The union of the retrieval channels, before the
    per-segment LLM filter runs."""
    return _apply("retrieved", segments, **kw)


def filtered(segments, **kw):
    """MODULES, refine. The filter's own verdict, before upstream's
    fallback."""
    return _apply("filtered", segments, **kw)
