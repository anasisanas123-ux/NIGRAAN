const Detection = require("../models/Detection");
const Unknown = require("../models/Unknown");

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

    // prevent duplicate logs — one per student per hour
    if (body.is_known && body.student_id) {
      const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
      const existing = await Detection.findOne({
        student_id: body.student_id,
        is_known: true,
        timestamp: { $gte: oneHourAgo },
      });
      if (existing) {
        return res.status(200).json({
          success: true,
          duplicate: true,
          message: "Already logged recently",
        });
      }
    }

    // prevent duplicate unknown logs — one per tracker per 10 minutes
    if (!body.is_known && body.tracker_id) {
      const tenMinAgo = new Date(Date.now() - 10 * 60 * 1000);
      const existing = await Detection.findOne({
        tracker_id: body.tracker_id,
        is_known: false,
        timestamp: { $gte: tenMinAgo },
      });
      if (existing) {
        return res.status(200).json({
          success: true,
          duplicate: true,
          message: "Unknown already logged recently",
        });
      }
    }

    const detection = await Detection.create(body);

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
    const io = req.app.get("io");
    if (io) io.emit("reset");
    res.json({ success: true, message: "All logs cleared" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
