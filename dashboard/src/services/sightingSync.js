// Copies the engine's sightings (GET /sightings?after=seq) into MongoDB every
// few seconds. Attendance and visitor history read from that collection.
const http = require('http');
const { ENGINE_PORT } = require('../paths');

function fetchSightings(after) {
  return new Promise((resolve, reject) => {
    const request = http.get({ hostname: '127.0.0.1', port: ENGINE_PORT(), path: `/sightings?after=${after}`, timeout: 3000 }, res => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(body).data); } catch (error) { reject(error); }
      });
    });
    request.on('timeout', () => request.destroy(new Error('timeout')));
    request.on('error', reject);
  });
}

function createSightingSync({ Sighting, io = null, fetch = fetchSightings, intervalMs = 5000, logger = console }) {
  let runId = null;
  let after = 0;
  let timer = null;

  async function poll() {
    let data;
    try { data = await fetch(after); } catch (_) { return 0; } // engine not running
    if (!data) return 0;
    if (data.run_id !== runId) {
      // The engine restarted (or this is the first poll): its numbering starts again.
      runId = data.run_id;
      after = 0;
      if (data.latest > 0) data = await fetch(0);
    }
    const events = data.events || [];
    if (!events.length) return 0;
    const docs = events.map(event => ({
      person_id: event.person_id, person_name: event.person_name, role: event.role,
      camera_id: event.camera_id, confidence: event.confidence,
      timestamp: new Date(event.timestamp), run_id: runId, seq: event.seq,
    }));
    try {
      await Sighting.insertMany(docs, { ordered: false });
    } catch (error) {
      // Duplicates (already stored before a dashboard restart) are expected.
      if (!(error && (error.code === 11000 || (error.writeErrors || []).every(e => (e.code ?? e.err?.code) === 11000)))) {
        logger.error(`[sightings] ${error.message}`);
      }
    }
    after = Math.max(after, ...events.map(e => e.seq));
    if (io) io.emit('sightings', docs.map(({ person_id, person_name, camera_id, timestamp }) => ({ person_id, person_name, camera_id, timestamp })));
    return docs.length;
  }

  function start() {
    if (timer) return;
    const loop = async () => {
      try { await poll(); } catch (error) { logger.error(`[sightings] ${error.message}`); }
      timer = setTimeout(loop, intervalMs);
    };
    timer = setTimeout(loop, 0);
  }

  return { start, stop: () => { clearTimeout(timer); timer = null; }, poll };
}

module.exports = { createSightingSync };
