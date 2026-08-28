const mongoose = require('mongoose');

const detectionSchema = new mongoose.Schema({
  camera_id: { type: String, required: true },
  camera_name: { type: String, default: null },

  // Stable application identity.
  // Known person  -> student/person ID
  // Unknown       -> global ReID ID such as UNK-001
  person_id: { type: String, default: null },
  global_person_id: { type: String, default: null },
  person_name: { type: String, default: 'Unknown' },

  role: {
    type: String,
    default: 'unknown',
    enum: ['student', 'faculty', 'staff', 'visitor', 'unknown'],
  },

  is_known: { type: Boolean, default: false },

  // Face-recognition confidence for known people.
  confidence: { type: Number, default: 0 },

  // Body Re-ID confidence for unknown people.
  reid_confidence: { type: Number, default: 0 },

  // DeepSORT ID belonging only to this camera.
  local_tracker_id: { type: String, default: null },

  // Kept so older dashboard code does not break.
  tracker_id: { type: String, default: null },

  event_type: {
    type: String,
    default: 'detected',
    enum: [
      'detected',
      'entered',
      'first_seen',
      'reidentified',
      'camera_transition',
      'reappeared',
    ],
  },

  previous_camera: { type: String, default: null },

  timestamp: { type: Date, default: Date.now },
  snapshot_path: { type: String, default: null },

  // Backward compatibility.
  student_id: { type: String, default: null },
  student_name: { type: String, default: 'Unknown' },
});

detectionSchema.index({ timestamp: -1 });
detectionSchema.index({ camera_id: 1, timestamp: -1 });
detectionSchema.index({ person_id: 1, timestamp: -1 });
detectionSchema.index({ global_person_id: 1, timestamp: -1 });
detectionSchema.index({ role: 1, timestamp: -1 });
detectionSchema.index({ event_type: 1, timestamp: -1 });

module.exports = mongoose.model('Detection', detectionSchema);