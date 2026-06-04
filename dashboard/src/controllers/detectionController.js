const Detection = require('../models/Detection');
const Unknown   = require('../models/Unknown');

exports.getRecentDetections = async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 50;
    const detections = await Detection.find()
      .sort({ timestamp: -1 })
      .limit(limit);
    res.json({ success: true, count: detections.length, data: detections });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getDetectionsByCamera = async (req, res) => {
  try {
    const detections = await Detection.find({ camera_id: req.params.camId })
      .sort({ timestamp: -1 })
      .limit(100);
    res.json({ success: true, count: detections.length, data: detections });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// Called by Python engine to log a detection
exports.ingestDetection = async (req, res) => {
  try {
    const detection = await Detection.create(req.body);

    // if unknown, update or create unknown tracker record
    if (!detection.is_known && detection.tracker_id) {
      await Unknown.findOneAndUpdate(
        { tracker_id: detection.tracker_id },
        {
          $set:  { last_seen: detection.timestamp, snapshot_path: detection.snapshot_path },
          $inc:  { detection_count: 1 },
          $addToSet: { cameras_seen: detection.camera_id }
        },
        { upsert: true, new: true }
      );
    }

    // broadcast to dashboard via Socket.io
    const io = req.app.get('io');
    if (io) io.emit('new_detection', detection);

    res.status(201).json({ success: true, data: detection });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

exports.getStats = async (req, res) => {
  try {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const [totalToday, knownToday, unknownToday, totalUnknowns] = await Promise.all([
      Detection.countDocuments({ timestamp: { $gte: today } }),
      Detection.countDocuments({ timestamp: { $gte: today }, is_known: true }),
      Detection.countDocuments({ timestamp: { $gte: today }, is_known: false }),
      Unknown.countDocuments({ is_resolved: false })
    ]);

    res.json({
      success: true,
      data: { totalToday, knownToday, unknownToday, totalUnknowns }
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};