const Visitor = require("../models/Visitor");
const path = require("path");
const fs = require("fs");

// directory where visitor photos and embeddings are stored
const VISITOR_PHOTOS_DIR = path.join(__dirname, "..", "..", "..", "data", "visitor_photos");
if (!fs.existsSync(VISITOR_PHOTOS_DIR)) {
  fs.mkdirSync(VISITOR_PHOTOS_DIR, { recursive: true });
}

function generateVisitorId() {
  return "V" + Date.now().toString().slice(-8);
}

exports.registerVisitor = async (req, res) => {
  try {
    const { name, contact, purpose, host_name, host_id, valid_hours, approved_by, photo_base64 } = req.body;

    if (!name || !photo_base64) {
      return res.status(400).json({ success: false, error: "name and photo are required" });
    }

    const visitor_id = generateVisitorId();
    const validHours = parseInt(valid_hours) || 4;
    const valid_until = new Date(Date.now() + validHours * 60 * 60 * 1000);

    // save photo to disk for the Python embedding builder to pick up
    const base64Data = photo_base64.replace(/^data:image\/\w+;base64,/, "");
    const photoFileName = `${visitor_id}.jpg`;
    const photoPath = path.join(VISITOR_PHOTOS_DIR, photoFileName);
    fs.writeFileSync(photoPath, base64Data, "base64");

    const visitor = await Visitor.create({
      visitor_id,
      name,
      contact: contact || null,
      purpose: purpose || null,
      host_name: host_name || null,
      host_id: host_id || null,
      photo_path: photoPath,
      valid_until,
      approved_by: approved_by || null,
      is_active: true,
    });

    const io = req.app.get("io");
    if (io) io.emit("visitor_registered", visitor);

    res.status(201).json({
      success: true,
      data: visitor,
      message: "Visitor registered. Run embedding builder to activate recognition.",
    });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

exports.getActiveVisitors = async (req, res) => {
  try {
    const now = new Date();
    const visitors = await Visitor.find({
      is_active: true,
      valid_until: { $gte: now },
    }).sort({ checked_in_at: -1 });

    res.json({ success: true, count: visitors.length, data: visitors });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getAllVisitors = async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const visitors = await Visitor.find().sort({ created_at: -1 }).limit(limit);
    res.json({ success: true, count: visitors.length, data: visitors });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.checkoutVisitor = async (req, res) => {
  try {
    const visitor = await Visitor.findOneAndUpdate(
      { visitor_id: req.params.visitorId },
      { is_active: false, checked_out_at: new Date() },
      { new: true }
    );

    if (!visitor) {
      return res.status(404).json({ success: false, error: "Visitor not found" });
    }

    const io = req.app.get("io");
    if (io) io.emit("visitor_checked_out", visitor);

    res.json({ success: true, data: visitor });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// called periodically to expire visitors whose valid_until has passed
exports.expireOldVisitors = async () => {
  try {
    const now = new Date();
    const result = await Visitor.updateMany(
      { is_active: true, valid_until: { $lt: now } },
      { is_active: false, checked_out_at: now }
    );
    if (result.modifiedCount > 0) {
      console.log(`Expired ${result.modifiedCount} visitor(s)`);
    }
  } catch (err) {
    console.error("Visitor expiry error:", err.message);
  }
};
