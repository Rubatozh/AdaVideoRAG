"""Seams the multimodal-RAG benchmark installs at runtime.

ADDED BY THE BENCHMARK. Not upstream. See bench/systems/ada.py.

Every hook DEFAULTS TO IDENTITY. With nothing installed this file changes no
behaviour at all, which is the point: the upstream execution path is not
"maintained alongside" the new one, it simply IS the path whenever the harness
has not installed a hook. There is no second branch to keep in sync and nothing
to bit-rot.

THE EDIT CONTRACT, so this stays revertible:

  * every inserted call site lives inside a `# BENCH SEAM BEGIN/END` block
  * the original expression is never modified, moved, or wrapped in a condition
  * deleting this file and every marker-delimited block restores the checkout
    byte-for-byte

`bench/tests/submodule_seams.py` asserts exactly that, so the fallback claim is
checked rather than promised.

WHY HOOKS FIRE IS RECORDED. `fired()` counts each hook's invocations and the
harness writes it onto every record. A hook that silently failed to install
looks identical to one that installed and did nothing -- that is how `vgent`
came to accept `--pool-rate`, name its arm for it, and ignore it.
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
    """{hook: n_calls}. Written onto the record so an arm proves its path."""
    return dict(_FIRED)


def _apply(name, value, **kw):
    fn = _HOOKS.get(name)
    if fn is None:
        return value                      # <- the upstream path, unchanged
    _FIRED[name] = _FIRED.get(name, 0) + 1
    return fn(value, **kw)


def _optional(name, **kw):
    """For hooks that produce something or decline. -> value or None.

    Distinct from `_apply` because these do not transform an existing value --
    there is nothing upstream to pass through, so "not installed" has to be
    expressible as None rather than as an identity.
    """
    fn = _HOOKS.get(name)
    if fn is None:
        return None                       # <- the upstream path, unchanged
    _FIRED[name] = _FIRED.get(name, 0) + 1
    return fn(**kw)


# --------------------------------------------------------------- the seams
def frame_times(times, **kw):
    """INPUT POOL, moments. The per-segment sample `deal_video` computed with
    its own `np.linspace`; the harness substitutes the pool's own timestamps so
    OCR, captioning and the stored `frame_times` all agree with what the pool
    planned."""
    return _apply("frame_times", times, **kw)


def frame_pixels(image, **kw):
    """INPUT POOL, pixels (single frame). `deal_video`'s OCR loop reads one
    frame at a time straight off the source clip; this returns the pool's
    JPEG for that moment instead, so OCR sees pool pixels rather than source
    pixels at a pool-adjacent timestamp."""
    return _apply("frame_pixels", image, **kw)


def frames_for(frames, **kw):
    """INPUT POOL, pixels (a segment's caption frames). `encode_video` is the
    single place the captioner decodes, so substituting here unifies both the
    moments and the pixels MiniCPM-V sees."""
    return _apply("frames_for", frames, **kw)


def segment_clip(**kw):
    """INPUT POOL, ImageBind. Write the segment's video file from POOL frames.

    ImageBind's `load_and_transform_video_data` takes a FILE and does its own
    clip sampling, so the segment mp4 must exist -- and upstream builds it by
    re-encoding the source, which is pixels the pool never authorised. The
    harness writes the same file from the pool's JPEGs instead, at a frame rate
    derived from the pool so the clip's DURATION is preserved (30 pool frames
    over a 30 s segment is a 1 fps clip, not a 1-second one).

    Returns True when it wrote the file. None/False means not handled, and
    upstream's own `write_videofile` runs exactly as before.
    """
    return _optional("segment_clip", **kw)


def retrieved(segments, **kw):
    """MODULES, retrieve. The union of the retrieval channels, BEFORE the
    per-segment LLM filter runs.

    Without this the benchmark can only observe the post-filter list, and
    upstream falls back to `remain = retrieved` when the filter rejects
    everything -- so a refine ablation could not tell "the filter kept all of
    them" from "the filter rejected all of them and the fallback fired".
    """
    return _apply("retrieved", segments, **kw)


def filtered(segments, **kw):
    """MODULES, refine. The filter's OWN verdict, BEFORE upstream's fallback.

    Upstream sets `remain = retrieved` whenever the filter rejects everything, so
    the list that reaches generation is identical in two opposite cases: the
    filter kept every candidate, or it rejected every candidate and the fallback
    fired. Only here are those distinguishable -- an empty value IS the fallback
    condition -- and they are opposite findings for a refine ablation.
    """
    return _apply("filtered", segments, **kw)
