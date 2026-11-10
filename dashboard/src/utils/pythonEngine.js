const http = require('http');

function requestEngine(method, endpoint, payload) {
  return new Promise((resolve, reject) => {
    const body = payload === undefined ? null : Buffer.from(JSON.stringify(payload));
    const request = http.request({
      hostname: '127.0.0.1',
      port: Number(process.env.PYTHON_ENGINE_PORT || 5001),
      path: endpoint,
      method,
      timeout: 60000,
      headers: body ? {
        'Content-Type': 'application/json',
        'Content-Length': body.length,
      } : {},
    }, (response) => {
      let raw = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { raw += chunk; });
      response.on('end', () => {
        let data;
        try { data = JSON.parse(raw); }
        catch (_) { return reject(new Error('Recognition engine returned an invalid response.')); }
        if (response.statusCode < 200 || response.statusCode >= 300 || !data.success) {
          const error = new Error(data.error || `Recognition engine request failed (${response.statusCode}).`);
          error.statusCode = response.statusCode;
          return reject(error);
        }
        resolve(data);
      });
    });
    request.on('timeout', () => request.destroy(new Error('Recognition engine request timed out.')));
    request.on('error', (error) => reject(new Error(`Recognition engine unavailable: ${error.message}`)));
    if (body) request.write(body);
    request.end();
  });
}

exports.enrollIdentity = (payload) => requestEngine('POST', '/enroll', payload);
exports.rebuildEmbeddings = () => requestEngine('POST', '/rebuild-embeddings', {});
// reason: "checked out" or "expired"; a removed visitor goes on the watch list.
exports.removeIdentity = (identityId, reason) => requestEngine('DELETE',
  `/identity/${encodeURIComponent(identityId)}${reason ? `?reason=${encodeURIComponent(reason)}` : ''}`);
exports.request = requestEngine;
