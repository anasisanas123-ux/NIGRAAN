const mongoose = require('mongoose');

const unknownSchema = new mongoose.Schema({
  // Stable ID shared between all cameras.
  global_id: {
    type: String,
    required: true,
    unique: true,
    sparse: true,
    index: true,
  },

  // Backward-compatible alias.
  tracker_id: {
    type: String,
    default: null,
  },

  first_seen: { type: Date, default: Date.now },
  last_seen: { type: Date, default: Date.now },

  last_camera: { type: String, default: null },

  cameras_seen: {
    type: [String],
    default: [],
  },

  // Example:
  // cam_a:12
  // cam_b:7
  //
  // These are local DeepSORT tracks that were linked
  // to this single global unknown identity.
  local_tracker_ids: {
    type: [String],
    default: [],
  },

  detection_count: {
    type: Number,
    default: 1,
  },

  snapshot_path: {
    type: String,
    default: null,
  },

  is_resolved: {
    type: Boolean,
    default: false,
  },
});

unknownSchema.index({ last_seen: -1 });
unknownSchema.index({ is_resolved: 1, last_seen: -1 });

module.exports = mongoose.model('Unknown', unknownSchema);