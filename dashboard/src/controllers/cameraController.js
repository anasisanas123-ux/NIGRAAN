const http = require('http');

exports.cameraZones = (req, res) => {
  const payload = req.method === 'PUT' ? JSON.stringify(req.body) : null;
  const upstream = http.request({
    hostname: '127.0.0.1', port: Number(process.env.PYTHON_ENGINE_PORT || 5001),
    path: `/zones/${encodeURIComponent(req.params.cameraId)}`, method: req.method,
    timeout: 5000, headers: payload ? {'Content-Type':'application/json', 'Content-Length':Buffer.byteLength(payload)} : {},
  }, response => {
    res.status(response.statusCode).set('Content-Type','application/json').set('Cache-Control','no-store');
    response.pipe(res);
    response.on('error', () => res.destroy());
  });
  upstream.on('timeout', () => upstream.destroy(new Error('timeout')));
  upstream.on('error', () => {
    if (!res.headersSent) res.status(503).json({success:false,error:'Camera engine is unavailable.'});
  });
  res.on('close', () => upstream.destroy());
  upstream.end(payload);
};

exports.getSnapshot = (req, res) => {
  const request = http.get({
    hostname: '127.0.0.1',
    port: Number(process.env.PYTHON_ENGINE_PORT || 5001),
    path: `/snapshot/${encodeURIComponent(req.params.cameraId)}`,
    timeout: 5000,
  }, (response) => {
    res.status(response.statusCode);
    res.set('Content-Type', response.headers['content-type'] || 'application/octet-stream');
    res.set('Cache-Control', 'no-store');
    response.pipe(res);
    response.on('error', () => res.destroy());
  });
  request.on('timeout', () => request.destroy(new Error('Camera capture timed out.')));
  request.on('error', () => {
    if (!res.headersSent) res.status(503).json({ success: false, error: 'Camera engine is unavailable. Start Python or choose Browser webcam.' });
  });
  res.on('close', () => request.destroy());
};

exports.getCameraHealth = (req, res) => {
  const port = Number(process.env.PYTHON_ENGINE_PORT || 5001);
  const request = http.get({
    hostname: '127.0.0.1',
    port,
    path: '/health',
    timeout: 2500,
  }, (response) => {
    let body = '';
    response.setEncoding('utf8');
    response.on('data', (chunk) => { body += chunk; });
    response.on('end', () => {
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return res.status(503).json({ success: false, error: 'Camera engine is unavailable.' });
      }
      try {
        const health = JSON.parse(body);
        return res.json({ success: true, data: health.cameras || {} });
      } catch (_) {
        return res.status(502).json({ success: false, error: 'Camera engine returned invalid health data.' });
      }
    });
  });

  request.on('timeout', () => request.destroy(new Error('timeout')));
  request.on('error', () => {
    if (!res.headersSent) {
      res.status(503).json({ success: false, error: 'Camera engine is not responding.' });
    }
  });
};

exports.getBehaviorAlerts = (req, res) => {
  const port = Number(process.env.PYTHON_ENGINE_PORT || 5001);
  const request = http.get({
    hostname: '127.0.0.1',
    port,
    path: '/behavior-alerts',
    timeout: 2500,
  }, (response) => {
    let body = '';
    response.setEncoding('utf8');
    response.on('data', (chunk) => { body += chunk; });
    response.on('end', () => {
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return res.status(503).json({ success: false, error: 'Camera engine is unavailable.' });
      }
      try {
        const result = JSON.parse(body);
        return res.json({
          success: true,
          data: result.alerts || [],
          model: result.violence_model || { state: 'unknown', error: null },
          weapon_model: result.weapon_model || { state: 'unknown', error: null },
          monitor: result.monitor || {},
        });
      } catch (_) {
        return res.status(502).json({ success: false, error: 'Camera engine returned invalid activity data.' });
      }
    });
  });

  request.on('timeout', () => request.destroy(new Error('timeout')));
  request.on('error', () => {
    if (!res.headersSent) {
      res.status(503).json({ success: false, error: 'Camera engine is not responding.' });
    }
  });
};
