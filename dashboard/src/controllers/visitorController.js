const Visitor = require("../models/Visitor");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const pythonEngine = require('../utils/pythonEngine');

// directory where visitor photos and embeddings are stored
const VISITOR_PHOTOS_DIR = path.join(__dirname, "..", "..", "..", "data", "visitor_photos");
if (!fs.existsSync(VISITOR_PHOTOS_DIR)) {
  fs.mkdirSync(VISITOR_PHOTOS_DIR, { recursive: true });
}

function generateVisitorId() {
  return `V${Date.now().toString(36).toUpperCase()}${crypto.randomBytes(3).toString("hex").toUpperCase()}`;
}

exports.registerVisitor = async (req, res) => {
  let photoPath;
  let tempPath;
  let enrolledIdentityId;
  try {
    const body = req.body || {};
    const name = typeof body.name === "string" ? body.name.trim() : "";
    const photoBase64 = typeof body.photo_base64 === "string" ? body.photo_base64 : "";

    if (!name || name.length > 120) {
      return res.status(400).json({ success: false, error: "Enter a visitor name up to 120 characters long." });
    }
    if (!photoBase64) {
      return res.status(400).json({ success: false, error: "A visitor photo is required." });
    }

    const photoMatch = photoBase64.match(/^data:image\/(jpeg|jpg|png|webp);base64,([A-Za-z0-9+/]+={0,2})$/i);
    if (!photoMatch) {
      return res.status(400).json({ success: false, error: "Upload a valid JPEG, PNG, or WebP visitor photo." });
    }
    const mime = photoMatch[1].toLowerCase();
    const imageBytes = Buffer.from(photoMatch[2], "base64");
    if (!imageBytes.length || imageBytes.length > 5 * 1024 * 1024) {
      return res.status(400).json({ success: false, error: "The photo must be smaller than 5 MB." });
    }
    const isJpeg = imageBytes[0] === 0xff && imageBytes[1] === 0xd8 && imageBytes[2] === 0xff;
    const isPng = imageBytes.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"));
    const isWebp = imageBytes.subarray(0, 4).toString() === "RIFF" && imageBytes.subarray(8, 12).toString() === "WEBP";
    if ((mime === "jpeg" || mime === "jpg") ? !isJpeg : mime === "png" ? !isPng : !isWebp) {
      return res.status(400).json({ success: false, error: "The uploaded photo data does not match its image format." });
    }

    const validHours = Number(body.valid_hours ?? 4);
    if (!Number.isInteger(validHours) || validHours < 1 || validHours > 24) {
      return res.status(400).json({ success: false, error: "Visit duration must be between 1 and 24 hours." });
    }

    const visitor_id = generateVisitorId();
    const valid_until = new Date(Date.now() + validHours * 60 * 60 * 1000);
    const extension = mime === "png" ? "png" : mime === "webp" ? "webp" : "jpg";
    photoPath = path.join(VISITOR_PHOTOS_DIR, `${visitor_id}.${extension}`);
    tempPath = `${photoPath}.${crypto.randomBytes(4).toString("hex")}.tmp`;
    await fs.promises.writeFile(tempPath, imageBytes, { flag: "wx" });
    await fs.promises.rename(tempPath, photoPath);
    tempPath = null;

    // Add the visitor's face to the running ArcFace gallery before the
    // dashboard reports success, so the live camera can recognize them now.
    await pythonEngine.enrollIdentity({
      identity_id: visitor_id,
      name,
      role: "visitor",
      valid_until: valid_until.toISOString(),
      images: [photoBase64],
    });
    enrolledIdentityId = visitor_id;

    const visitor = await Visitor.create({
      visitor_id,
      name,
      contact: typeof body.contact === "string" ? body.contact.trim().slice(0, 80) || null : null,
      purpose: typeof body.purpose === "string" ? body.purpose.trim().slice(0, 240) || null : null,
      host_name: typeof body.host_name === "string" ? body.host_name.trim().slice(0, 120) || null : null,
      host_id: typeof body.host_id === "string" ? body.host_id.trim().slice(0, 80) || null : null,
      photo_path: photoPath,
      valid_until,
      approved_by: typeof body.approved_by === "string" ? body.approved_by.trim().slice(0, 120) || null : null,
      is_active: true,
    });

    const io = req.app.get("io");
    if (io) io.emit("visitor_registered", visitor);
    enrolledIdentityId = null;

    res.status(201).json({
      success: true,
      data: visitor,
      message: "Visitor registered and the photo was saved.",
    });
  } catch (err) {
    if (enrolledIdentityId) {
      try { await pythonEngine.removeIdentity(enrolledIdentityId); } catch (_) { /* a later engine restart will reconcile the gallery */ }
    }
    if (tempPath) {
      try { await fs.promises.unlink(tempPath); } catch (_) { /* no temp photo remains */ }
    }
    if (photoPath) {
      try { await fs.promises.unlink(photoPath); } catch (_) { /* no photo was committed */ }
    }
    const status = err && err.code === 11000 ? 409 : (err && err.name === "ValidationError" ? 400 : err.statusCode === 400 ? 400 : err.message?.startsWith('Recognition engine') ? 503 : 500);
    res.status(status).json({ success: false, error: err.message || "Could not register visitor." });
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
    const visitor = await Visitor.findOne({ visitor_id: req.params.visitorId });
    if (!visitor) {
      return res.status(404).json({ success: false, error: "Visitor not found" });
    }
    try {
      await pythonEngine.removeIdentity(visitor.visitor_id);
    } catch (error) {
      return res.status(503).json({ success: false, error: `Visitor could not be removed from live recognition yet: ${error.message}` });
    }
    visitor.is_active = false;
    visitor.checked_out_at = new Date();
    await visitor.save();

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
    const expired = await Visitor.find({ is_active: true, valid_until: { $lt: now } }).select('visitor_id').lean();
    const result = await Visitor.updateMany(
      { is_active: true, valid_until: { $lt: now } },
      { is_active: false, checked_out_at: now }
    );
    if (result.modifiedCount > 0) {
      console.log(`Expired ${result.modifiedCount} visitor(s)`);
      for (const visitor of expired) {
        try { await pythonEngine.removeIdentity(visitor.visitor_id); }
        catch (error) { console.warn(`Could not refresh expired visitor in camera gallery (${visitor.visitor_id}): ${error.message}`); }
      }
    }
  } catch (err) {
    console.error("Visitor expiry error:", err.message);
  }
};
