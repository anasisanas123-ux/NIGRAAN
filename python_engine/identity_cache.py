"""Keep local tracking caches consistent with live enrollment and removal."""


def invalidate_inactive(recognizer, confirmed, locked):
    for cache in (confirmed, locked):
        for key, entry in list(cache.items()):
            identity = entry.get('student_id')
            if identity is not None and not recognizer.is_identity_active(identity):
                cache.pop(key, None)


def needs_face_check(entry, gallery_version, now, interval):
    if not entry or entry.get('global_id') is None:
        return True
    return (entry.get('gallery_version', -1) != gallery_version
            or now - entry.get('last_face_check', -float('inf')) >= interval)


def update_pending_hold(holds, key, face_pending, now, seconds):
    """Return whether a track is confirming an enrolled face.

    face_pending is True or False when a face was evaluated this frame, and
    None for a body-only frame, which only inherits a hold that has not expired.
    """
    for held_key, until in list(holds.items()):
        if until <= now:
            holds.pop(held_key, None)
    if face_pending is None:
        return key in holds
    if face_pending:
        holds[key] = now + seconds
        return True
    holds.pop(key, None)
    return False


def is_rejected_weak_detection(global_id, face_embedding, body_confidence, minimum):
    """True for a body-only detection too weak to be treated as a person."""
    return global_id is None and face_embedding is None and body_confidence < minimum
