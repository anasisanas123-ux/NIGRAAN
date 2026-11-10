const mongoose = require('mongoose');

// A recognised person seen on a camera. The engine logs each person at most
// once per camera every SIGHTING_INTERVAL_SECONDS (30 s by default).
const sightingSchema = new mongoose.Schema({
  person_id:   { type: String, required: true },
  person_name: { type: String, default: null },
  role:        { type: String, default: 'student' },
  camera_id:   { type: String, required: true },
  confidence:  { type: Number, default: 0 },
  timestamp:   { type: Date, required: true },
  // Engine run + sequence number, so a re-read never stores a sighting twice.
  run_id:      { type: String, required: true },
  seq:         { type: Number, required: true },
});

sightingSchema.index({ run_id: 1, seq: 1 }, { unique: true });
sightingSchema.index({ camera_id: 1, timestamp: 1 });
sightingSchema.index({ person_id: 1, timestamp: -1 });

module.exports = mongoose.model('Sighting', sightingSchema);
