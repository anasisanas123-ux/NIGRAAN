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
