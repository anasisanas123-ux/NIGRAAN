const mongoose = require('mongoose');

const unknownSchema = new mongoose.Schema({
  tracker_id:   { type: String, required: true, unique: true },
  first_seen:   { type: Date, default: Date.now },
  last_seen:    { type: Date, default: Date.now },
  cameras_seen: { type: [String], default: [] },
  detection_count: { type: Number, default: 1 },
  snapshot_path:   { type: String, default: null },
  is_resolved:     { type: Boolean, default: false }
});

module.exports = mongoose.model('Unknown', unknownSchema);