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


def contested_track_keys(boxes, min_overlap, identities=None):
    """Keys whose box overlaps another box by at least min_overlap.

    Overlap is intersection over the smaller box, so a person passing in front
    of a smaller (seated or distant) person counts even when IoU is low.
    Two boxes already holding the same identity are duplicate tracks of one
    person, not two people, so they do not contest each other.
    """
    identities = identities or {}
    keys = list(boxes)
    contested = set()
    for index, first in enumerate(keys):
        ax1, ay1, ax2, ay2 = boxes[first]
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        for second in keys[index + 1:]:
            identity = identities.get(first)
            if identity is not None and identity == identities.get(second):
                continue
            bx1, by1, bx2, by2 = boxes[second]
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            smaller = min(area_a, area_b)
            if smaller <= 0:
                continue
            width = max(0, min(ax2, bx2) - max(ax1, bx1))
            height = max(0, min(ay2, by2) - max(ay1, by1))
            if width * height / smaller >= min_overlap:
                contested.update((first, second))
    return contested


def is_rejected_weak_detection(global_id, face_embedding, body_confidence, minimum):
    """True for a body-only detection too weak to be treated as a person."""
    return global_id is None and face_embedding is None and body_confidence < minimum
