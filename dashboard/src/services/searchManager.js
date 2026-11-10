// Appearance search: lists searchable videos (recordings + uploads), starts the
// Python search service on demand, queues indexing and runs queries.
const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const { spawn } = require('child_process');
const P = require('../paths');

const VIDEO_TYPES = new Set(['.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm']);
const SAFE_ID = /^[A-Za-z0-9_.-]{1,120}$/;

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (_) { return null; }
}

function writeJson(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(data, null, 2));
  fs.renameSync(temp, file);
}

function indexFolder(sourceId) {
  if (!SAFE_ID.test(sourceId)) throw new Error('Invalid video ID.');
  return path.join(P.SEARCH_INDEX_DIR, sourceId);
}

function indexState(sourceId) {
  const folder = indexFolder(sourceId);
  const status = readJson(path.join(folder, 'status.json')) || {};
  if (fs.existsSync(path.join(folder, 'tracks.json')) && status.state !== 'running' && status.state !== 'queued') {
    return { state: 'indexed', people: status.people ?? null, indexed_at: status.indexed_at || null };
  }
  return { state: status.state || 'not_indexed', progress: status.progress || 0, error: status.error || null };
}

function listRecordings() {
  const out = [];
  if (!fs.existsSync(P.RECORDINGS_DIR)) return out;
  for (const camera of fs.readdirSync(P.RECORDINGS_DIR)) {
    const folder = path.join(P.RECORDINGS_DIR, camera);
    if (!fs.statSync(folder).isDirectory()) continue;
    for (const name of fs.readdirSync(folder)) {
      if (!name.endsWith('.json')) continue;
      const meta = readJson(path.join(folder, name));
      if (!meta || !meta.file || !SAFE_ID.test(meta.id || '')) continue;
      const video = path.join(P.RECORDINGS_DIR, meta.file);
      if (!fs.existsSync(video)) continue;
      const id = `rec-${meta.id}`;
      out.push({
        id, kind: 'recording', name: meta.label || `${meta.camera_id} recording`,
        camera_id: meta.camera_id, started_at: meta.started_at, ended_at: meta.ended_at,
        duration_seconds: meta.duration_seconds, status: meta.status,
        size_bytes: fs.statSync(video).size, video_path: video,
        url: `/media/recordings/${meta.file}`, index: indexState(id),
      });
    }
  }
  return out.sort((a, b) => String(b.started_at).localeCompare(String(a.started_at)));
}

function listUploads() {
  const out = [];
  if (!fs.existsSync(P.UPLOADS_DIR)) return out;
  for (const name of fs.readdirSync(P.UPLOADS_DIR)) {
    if (!name.endsWith('.json')) continue;
    const meta = readJson(path.join(P.UPLOADS_DIR, name));
    if (!meta || !SAFE_ID.test(meta.id || '') || !meta.file) continue;
    const video = path.join(P.UPLOADS_DIR, meta.file);
    if (!fs.existsSync(video)) continue;
    out.push({
      id: meta.id, kind: 'upload', name: meta.original_name, uploaded_at: meta.uploaded_at,
      size_bytes: meta.size_bytes, video_path: video, url: `/media/uploads/${meta.file}`,
      index: indexState(meta.id),
    });
  }
  return out.sort((a, b) => String(b.uploaded_at).localeCompare(String(a.uploaded_at)));
}

function listSources() {
  return [...listRecordings(), ...listUploads()];
}

function publicSource(source) {
  const { video_path, ...rest } = source;
  return rest;
}

// ---------------------------------------------------------------- service
function request(method, endpoint, body, timeout = 120000) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const req = http.request({
      hostname: '127.0.0.1', port: P.SEARCH_PORT(), path: endpoint, method, timeout,
      headers: payload ? { 'Content-Type': 'application/json', 'Content-Length': payload.length } : {},
    }, res => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { raw += chunk; });
      res.on('end', () => {
        let data;
        try { data = JSON.parse(raw); } catch (_) { return reject(new Error('Search service returned an invalid response.')); }
        if (res.statusCode >= 400) return reject(Object.assign(new Error(data.error || `Search service error ${res.statusCode}`), { statusCode: res.statusCode }));
        resolve(data);
      });
    });
    req.on('timeout', () => req.destroy(new Error('Search service timed out.')));
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

function createSearchManager({ io = null, logger = console } = {}) {
  let child = null;
  let starting = null;
  const watching = new Map(); // sourceId -> jobId

  async function healthy() {
    try { return (await request('GET', '/health', undefined, 2000)).ready === true; } catch (_) { return false; }
  }

  function ensureService() {
    if (starting) return starting;
    starting = (async () => {
      if (await healthy()) return true;
      if (!child) {
        const python = process.platform === 'win32'
          ? path.join(P.ROOT_DIR, 'venv', 'Scripts', 'python.exe')
          : path.join(P.ROOT_DIR, 'venv', 'bin', 'python');
        const logDir = path.join(P.DATA_DIR, 'logs');
        fs.mkdirSync(logDir, { recursive: true });
        const log = fs.openSync(path.join(logDir, 'search_service.log'), 'a');
        child = spawn(python, ['search_service.py'], {
          cwd: P.ROOT_DIR, stdio: ['ignore', log, log], windowsHide: true,
          env: { ...process.env, SEARCH_SERVICE_PORT: String(P.SEARCH_PORT()) },
        });
        child.on('exit', code => {
          logger.warn(`Appearance search service stopped (exit ${code}). See data/logs/search_service.log`);
          child = null;
        });
        logger.log('Starting appearance search service (first start takes about 15 s)...');
      }
      for (let i = 0; i < 180; i++) {
        if (await healthy()) return true;
        if (!child) break;
        await new Promise(r => setTimeout(r, 1000));
      }
      throw new Error('The appearance search service did not start. See data/logs/search_service.log.');
    })();
    starting.finally(() => { starting = null; }).catch(() => {});
    return starting;
  }

  function stopService() {
    if (child) { try { child.kill(); } catch (_) { /* already gone */ } }
  }

  function setStatus(sourceId, status) {
    writeJson(path.join(indexFolder(sourceId), 'status.json'), status);
    if (io) io.emit('search_index', { source_id: sourceId, ...status });
  }

  async function watch(sourceId, jobId) {
    watching.set(sourceId, jobId);
    try {
      for (;;) {
        await new Promise(r => setTimeout(r, 1000));
        if (watching.get(sourceId) !== jobId) return;
        let job;
        try { job = await request('GET', `/jobs/${encodeURIComponent(jobId)}`, undefined, 5000); }
        catch (error) {
          if (!child && !(await healthy())) { setStatus(sourceId, { state: 'error', error: 'The search service stopped.' }); return; }
          continue;
        }
        if (job.state === 'done') {
          setStatus(sourceId, { state: 'indexed', progress: 1, people: job.result?.people ?? null,
            indexed_at: new Date().toISOString(), seconds_taken: job.result?.seconds_taken });
          return;
        }
        if (job.state === 'error') { setStatus(sourceId, { state: 'error', error: job.error }); return; }
        setStatus(sourceId, { state: job.state, progress: job.progress || 0 });
      }
    } finally {
      if (watching.get(sourceId) === jobId) watching.delete(sourceId);
    }
  }

  async function index(sourceIds, { force = false } = {}) {
    const sources = new Map(listSources().map(s => [s.id, s]));
    const chosen = sourceIds.map(id => sources.get(id)).filter(Boolean);
    if (!chosen.length) throw Object.assign(new Error('Choose at least one video.'), { statusCode: 400 });
    await ensureService();
    const queued = [];
    for (const source of chosen) {
      if (watching.has(source.id)) continue;
      if (!force && source.index.state === 'indexed') continue;
      const folder = indexFolder(source.id);
      if (force) fs.rmSync(folder, { recursive: true, force: true });
      fs.mkdirSync(folder, { recursive: true });
      const jobId = `${source.id}-${Date.now()}`;
      setStatus(source.id, { state: 'queued', progress: 0 });
      await request('POST', '/index', { job_id: jobId, video: source.video_path, out: folder });
      queued.push(source.id);
      watch(source.id, jobId).catch(error => logger.error(`[search] ${error.message}`));
    }
    return queued;
  }

  async function query({ sourceIds, text, photo, upper, lower }) {
    const all = new Map(listSources().map(s => [s.id, s]));
    const chosen = (sourceIds?.length ? sourceIds : [...all.keys()]).map(id => all.get(id))
      .filter(s => s && s.index.state === 'indexed');
    if (!chosen.length) throw Object.assign(new Error('None of the chosen videos is indexed yet. Index them first.'), { statusCode: 400 });
    if (!String(text || '').trim() && !photo && !upper && !lower) {
      throw Object.assign(new Error('Type a description, add a photo, or pick a clothing colour.'), { statusCode: 400 });
    }
    await ensureService();
    let photoPath = null;
    try {
      if (photo) {
        const match = /^data:image\/(jpeg|png|webp);base64,([A-Za-z0-9+/=]+)$/.exec(String(photo));
        if (!match) throw Object.assign(new Error('Use a JPEG, PNG or WebP photo.'), { statusCode: 400 });
        const bytes = Buffer.from(match[2], 'base64');
        if (bytes.length > 6 * 1024 * 1024) throw Object.assign(new Error('The photo must be under 6 MB.'), { statusCode: 400 });
        fs.mkdirSync(P.SEARCH_QUERY_DIR, { recursive: true });
        photoPath = path.join(P.SEARCH_QUERY_DIR, `${crypto.randomUUID()}.${match[1] === 'jpeg' ? 'jpg' : match[1]}`);
        fs.writeFileSync(photoPath, bytes);
      }
      const result = await request('POST', '/search', {
        sources: chosen.map(s => ({ id: s.id, folder: indexFolder(s.id) })),
        text: String(text || '').slice(0, 200), photo: photoPath, upper: upper || null, lower: lower || null, limit: 60,
      });
      const byId = new Map(chosen.map(s => [s.id, s]));
      const matches = result.matches.map(m => {
        const source = byId.get(m.source_id);
        const base = `/media/index/${m.source_id}/`;
        const startedAt = source.kind === 'recording' && source.started_at ? new Date(source.started_at) : null;
        return {
          ...m, source_name: source.name, source_kind: source.kind, camera_id: source.camera_id || null,
          video_url: source.url, thumb_url: base + m.thumb, scene_url: base + m.scene,
          wall_time: startedAt ? new Date(startedAt.getTime() + m.best_time * 1000).toISOString() : null,
        };
      });
      return { matches, face_in_photo: result.face_in_photo, searched: chosen.length };
    } finally {
      if (photoPath) fs.rm(photoPath, { force: true }, () => {});
    }
  }

  function remove(sourceId) {
    const source = listSources().find(s => s.id === sourceId);
    if (!source) throw Object.assign(new Error('Video not found.'), { statusCode: 404 });
    if (source.status === 'recording') throw Object.assign(new Error('Stop the recording before deleting it.'), { statusCode: 409 });
    watching.delete(sourceId);
    fs.rmSync(indexFolder(sourceId), { recursive: true, force: true });
    fs.rmSync(source.video_path, { force: true });
    const sidecar = source.kind === 'recording'
      ? source.video_path.replace(/\.mp4$/i, '.json')
      : path.join(P.UPLOADS_DIR, `${source.id}.json`);
    fs.rmSync(sidecar, { force: true });
    return true;
  }

  return { listSources: () => listSources().map(publicSource), index, query, remove, ensureService, stopService, healthy };
}

module.exports = { createSearchManager, VIDEO_TYPES, SAFE_ID, indexFolder };
