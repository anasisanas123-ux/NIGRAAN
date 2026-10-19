require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const express  = require('express');
const http     = require('http');
const socketio = require('socket.io');
const mongoose = require('mongoose');
const cors     = require('cors');
const path     = require('path');

const studentRoutes   = require('./routes/students');
const detectionRoutes = require('./routes/detections');
const unknownRoutes   = require('./routes/unknowns');
const visitorRoutes   = require('./routes/visitors');
const cameraRoutes    = require('./routes/cameras');

const app    = express();
const server = http.createServer(app);
const io     = socketio(server, { cors: { origin: '*' } });

app.set('io', io);

app.use(cors());
// Visitor photos are sent as base64 JSON. Allow requests large enough for
// compressed images while keeping a finite request-size limit.
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, '../public')));

// routes
app.use('/api/students',   studentRoutes);
app.use('/api/detections', detectionRoutes);
app.use('/api/unknowns',   unknownRoutes);
app.use('/api/visitors',   visitorRoutes);
app.use('/api/cameras',    cameraRoutes);

// health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date() });
});

// video feed proxy route
app.get('/video_feed/:cam_id', (req, res) => {
  const camId  = req.params.cam_id;
  const http_  = require('http');
  const url    = `http://127.0.0.1:${process.env.PYTHON_ENGINE_PORT || 5001}/video_feed/${encodeURIComponent(camId)}`;

  const proxyReq = http_.get(url, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
    proxyRes.on('error', () => res.destroy());
  });

  proxyReq.on('error', () => {
    if (!res.headersSent) res.status(503).json({ error: 'Python engine not running' });
    else res.destroy();
  });
  res.on('close', () => proxyReq.destroy());
});

// cross camera alert endpoint — called by Python
app.post('/api/cross_camera', (req, res) => {
  const data = req.body;
  io.emit('cross_camera', data);
  res.json({ success: true });
});

// serve dashboard
app.get('/{*path}', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// socket connection log
io.on('connection', (socket) => {
  console.log(`Dashboard client connected: ${socket.id}`);
  socket.on('disconnect', () => {
    console.log(`Dashboard client disconnected: ${socket.id}`);
  });
});

// connect MongoDB then start server
const PORT     = process.env.PORT || 3000;
const MONGO_URI = process.env.MONGODB_URI;

mongoose.connect(MONGO_URI)
  .then(() => {
    console.log('MongoDB connected');
    server.listen(PORT, () => {
      console.log(`\nServer running at http://localhost:${PORT}`);

      // periodically expire visitors whose valid_until time has passed
      const { expireOldVisitors } = require('./controllers/visitorController');
      setInterval(expireOldVisitors, 5 * 60 * 1000); // every 5 minutes
      expireOldVisitors(); // run once immediately on startup

      console.log('API endpoints:');
      console.log(`  GET  http://localhost:${PORT}/api/health`);
      console.log(`  GET  http://localhost:${PORT}/api/students`);
      console.log(`  GET  http://localhost:${PORT}/api/detections`);
      console.log(`  GET  http://localhost:${PORT}/api/unknowns`);
      console.log(`  GET  http://localhost:${PORT}/video_feed/cam_a`);
      console.log(`  GET  http://localhost:${PORT}/video_feed/cam_b`);
    });
  })
  .catch(err => {
    console.error('MongoDB connection failed:', err.message);
    process.exit(1);
  });
