const mongoose = require('mongoose');

const detectionSchema = new mongoose.Schema({
  camera_id:    { type: String, required: true },
  student_id:   { type: String, default: null },
  student_name: { type: String, default: 'Unknown' },
  is_known:     { type: Boolean, default: false },
  confidence:   { type: Number, default: 0 },
  tracker_id:   { type: String, default: null },
  timestamp:    { type: Date, default: Date.now },
  snapshot_path: { type: String, default: null }
});

detectionSchema.index({ timestamp: -1 });
detectionSchema.index({ camera_id: 1 });
detectionSchema.index({ student_id: 1 });

module.exports = mongoose.model('Detection', detectionSchema);