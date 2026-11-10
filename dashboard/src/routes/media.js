// Recordings, uploaded videos, appearance search, exam mode and the media files
// they produce (videos, search thumbnails, exam snapshots).
const express = require('express');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const P = require('../paths');
const engine = require('../utils/pythonEngine');
const { VIDEO_TYPES } = require('../services/searchManager');

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_MB || 4096) * 1024 * 1024;
const CAMERA = /^[A-Za-z0-9_-]{1,40}$/;

function fail(res, error) {
  const status = error.statusCode && error.statusCode < 600 ? error.statusCode
    : /unavailable|timed out|ECONNREFUSED/i.test(error.message) ? 503 : 500;
  res.status(status).json({ success: false, error: error.message });
}

module.exports = function mediaRoutes(searchManager) {
  const router = express.Router();

  // ---------------- recordings (start/stop live in the engine)
  router.get('/recordings/status', async (req, res) => {
    try { res.json({ success: true, data: (await engine.request('GET', '/recordings/status')).data }); }
    catch (e) { fail(res, e); }
  });
  router.post('/recordings/:camera/:action', async (req, res) => {
    if (!['start', 'stop'].includes(req.params.action)) return res.status(404).json({ success: false, error: 'Unknown action.' });
    if (!CAMERA.test(req.params.camera)) return res.status(400).json({ success: false, error: 'Invalid camera.' });
    try {
      const body = req.params.action === 'start' ? { label: String(req.body?.label || '').slice(0, 80) } : {};
      const result = await engine.request('POST', `/recordings/${req.params.camera}/${req.params.action}`, body);
      req.app.get('io')?.emit('recordings_changed');
      res.json({ success: true, data: result.data });
    } catch (e) { fail(res, e); }
  });

  // ---------------- searchable videos
  router.get('/search/sources', (req, res) => {
    try { res.json({ success: true, data: searchManager.listSources() }); } catch (e) { fail(res, e); }
  });
  router.delete('/search/sources/:id', (req, res) => {
    try { searchManager.remove(req.params.id); res.json({ success: true }); } catch (e) { fail(res, e); }
  });
  router.post('/search/index', async (req, res) => {
    try {
      const ids = Array.isArray(req.body?.source_ids) ? req.body.source_ids.map(String) : [];
      res.json({ success: true, data: { queued: await searchManager.index(ids, { force: Boolean(req.body?.force) }) } });
    } catch (e) { fail(res, e); }
  });
  router.post('/search/query', async (req, res) => {
    try {
      const b = req.body || {};
      res.json({ success: true, data: await searchManager.query({
        sourceIds: Array.isArray(b.source_ids) ? b.source_ids.map(String) : [],
        text: b.text, photo: b.photo, upper: b.upper, lower: b.lower,
      }) });
    } catch (e) { fail(res, e); }
  });

  // Upload a video: the raw file is the request body (streamed, not JSON).
  router.post('/uploads', (req, res) => {
    const original = decodeURIComponent(String(req.get('x-file-name') || 'video.mp4')).replace(/[\\/\x00-\x1f]/g, '_').slice(0, 120);
    const ext = path.extname(original).toLowerCase();
    if (!VIDEO_TYPES.has(ext)) return res.status(400).json({ success: false, error: 'Upload an MP4, MOV, AVI, MKV, M4V or WebM video.' });
    const declared = Number(req.get('content-length') || 0);
    if (declared > MAX_UPLOAD_BYTES) return res.status(413).json({ success: false, error: `The video is larger than ${Math.round(MAX_UPLOAD_BYTES / 1048576)} MB.` });
    fs.mkdirSync(P.UPLOADS_DIR, { recursive: true });
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
    const id = `up-${stamp}-${crypto.randomBytes(3).toString('hex')}`;
    const file = `${id}${ext}`;
    const target = path.join(P.UPLOADS_DIR, file);
    const out = fs.createWriteStream(target, { flags: 'wx' });
    let size = 0;
    let failed = false;
    const abort = (status, message) => {
      if (failed) return;
      failed = true;
      out.destroy();
      fs.rm(target, { force: true }, () => {});
      if (!res.headersSent) res.status(status).json({ success: false, error: message });
    };
    req.on('data', chunk => {
      size += chunk.length;
      if (size > MAX_UPLOAD_BYTES) { abort(413, 'The video is too large.'); req.destroy(); }
    });
    req.on('aborted', () => abort(400, 'Upload was interrupted.'));
    out.on('error', () => abort(500, 'Could not save the video.'));
    out.on('finish', () => {
      if (failed) return;
      if (size < 1024) return abort(400, 'The file is empty.');
      const meta = { id, file, original_name: original, size_bytes: size, uploaded_at: new Date().toISOString() };
      fs.writeFileSync(path.join(P.UPLOADS_DIR, `${id}.json`), JSON.stringify(meta, null, 2));
      res.status(201).json({ success: true, data: meta });
    });
    req.pipe(out);
  });

  // ---------------- exam mode (runs in the engine)
  router.get('/exams/status', async (req, res) => {
    try { res.json({ success: true, data: (await engine.request('GET', '/exam/status')).data }); } catch (e) { fail(res, e); }
  });
  router.get('/exams/reports', async (req, res) => {
    try { res.json({ success: true, data: (await engine.request('GET', '/exam/reports')).data }); } catch (e) { fail(res, e); }
  });
  router.get('/exams/reports/:id', async (req, res) => {
    if (!/^[A-Za-z0-9_-]{1,80}$/.test(req.params.id)) return res.status(400).json({ success: false, error: 'Invalid report.' });
    try { res.json({ success: true, data: (await engine.request('GET', `/exam/reports/${req.params.id}`)).data }); } catch (e) { fail(res, e); }
  });
  router.post('/exams/:camera/:action', async (req, res) => {
    if (!['start', 'stop', 'relearn'].includes(req.params.action)) return res.status(404).json({ success: false, error: 'Unknown action.' });
    if (!CAMERA.test(req.params.camera)) return res.status(400).json({ success: false, error: 'Invalid camera.' });
    try {
      const body = req.params.action === 'start' ? { name: String(req.body?.name || '').slice(0, 100) } : {};
      const result = await engine.request('POST', `/exam/${req.params.camera}/${req.params.action}`, body);
      req.app.get('io')?.emit('exam_changed');
      res.json({ success: true, data: result.data });
    } catch (e) { fail(res, e); }
  });

  return router;
};

// Static media with a type whitelist and a containment check. sendFile
// supports HTTP range requests, so videos can be seeked in the browser.
module.exports.mediaHandler = function mediaHandler(req, res) {
  const roots = { recordings: P.RECORDINGS_DIR, uploads: P.UPLOADS_DIR, index: P.SEARCH_INDEX_DIR, exams: P.EXAMS_DIR };
  const root = roots[req.params.kind];
  const rest = Array.isArray(req.params.rest) ? req.params.rest.join('/') : String(req.params.rest || '');
  const ext = path.extname(rest).toLowerCase();
  const allowed = req.params.kind === 'index' || req.params.kind === 'exams' ? new Set(['.jpg']) : VIDEO_TYPES;
  if (!root || !allowed.has(ext)) return res.status(404).end();
  const target = path.resolve(root, rest);
  if (!target.startsWith(path.resolve(root) + path.sep)) return res.status(404).end();
  res.sendFile(target, { headers: { 'Cache-Control': 'no-store' } }, error => {
    if (error && !res.headersSent) res.status(404).end();
  });
};
