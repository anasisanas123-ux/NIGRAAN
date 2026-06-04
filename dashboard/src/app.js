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

const app    = express();
const server = http.createServer(app);
const io     = socketio(server, { cors: { origin: '*' } });

app.set('io', io);

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// routes
app.use('/api/students',   studentRoutes);
app.use('/api/detections', detectionRoutes);
app.use('/api/unknowns',   unknownRoutes);

// health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date() });
});

// video feed proxy route
app.get('/video_feed/:cam_id', (req, res) => {
  const camId  = req.params.cam_id;
  const http_  = require('http');
  const url    = `http://localhost:${process.env.PYTHON_ENGINE_PORT || 5001}/video_feed/${camId}`;

  const proxyReq = http_.get(url, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on('error', () => {
    res.status(503).json({ error: 'Python engine not running' });
  });
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
    console.log('MongoDB connected:', MONGO_URI);
    server.listen(PORT, () => {
      console.log(`\nServer running at http://localhost:${PORT}`);
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