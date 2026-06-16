const Detection = require("../models/Detection");
const Unknown = require("../models/Unknown");

// in-memory session log — one entry per student per session
// resets when Node.js restarts (which is what we want)
const sessionLogged = new Set(); // known students
const unknownLogged = new Set(); // unknown tracker IDs

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

exports.ingestDetection = async (req, res) => {
  try {
    const body = req.body;

    // known student — log only once per session
    if (body.is_known && body.student_id) {
      const key = `${body.student_id}_${body.camera_id}`;
      if (sessionLogged.has(key)) {
        return res.status(200).json({
          success: true,
          duplicate: true,
          message: "Already logged this session",
        });
      }
      sessionLogged.add(key);
    }

    // unknown person — log only once per tracker ID per session
    if (!body.is_known && body.tracker_id) {
      if (unknownLogged.has(body.tracker_id)) {
        return res.status(200).json({
          success: true,
          duplicate: true,
          message: "Unknown already logged this session",
        });
      }
      unknownLogged.add(body.tracker_id);
    }

    // save to MongoDB
    const detection = await Detection.create(body);

    // update unknowns collection
    if (!detection.is_known && detection.tracker_id) {
      await Unknown.findOneAndUpdate(
        { tracker_id: detection.tracker_id },
        {
          $set: {
            last_seen: detection.timestamp,
            snapshot_path: detection.snapshot_path,
          },
          $inc: { detection_count: 1 },
          $addToSet: { cameras_seen: detection.camera_id },
        },
        { upsert: true, new: true },
      );
    }

    // emit instantly via Socket.io
    const io = req.app.get("io");
    if (io) io.emit("new_detection", detection);

    res.status(201).json({ success: true, data: detection });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

exports.getStats = async (req, res) => {
  try {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const [totalToday, knownToday, unknownToday, totalUnknowns] =
      await Promise.all([
        Detection.countDocuments({ timestamp: { $gte: today } }),
        Detection.countDocuments({
          timestamp: { $gte: today },
          is_known: true,
        }),
        Detection.countDocuments({
          timestamp: { $gte: today },
          is_known: false,
        }),
        Unknown.countDocuments({ is_resolved: false }),
      ]);

    res.json({
      success: true,
      data: { totalToday, knownToday, unknownToday, totalUnknowns },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.resetDetections = async (req, res) => {
  try {
    await Detection.deleteMany({});
    await Unknown.deleteMany({});

    // clear in-memory session logs on reset
    sessionLogged.clear();
    unknownLogged.clear();

    const io = req.app.get("io");
    if (io) io.emit("reset");

    res.json({ success: true, message: "All logs cleared" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
