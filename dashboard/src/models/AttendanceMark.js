const mongoose = require('mongoose');

// A manual correction by staff, e.g. "excused" or "present" for a student the
// camera missed. It overrides the automatic status for that class and date.
const markSchema = new mongoose.Schema({
  class_id:   { type: String, required: true },
  date:       { type: String, required: true, match: /^\d{4}-\d{2}-\d{2}$/ },
  student_id: { type: String, required: true },
  status:     { type: String, required: true, enum: ['present', 'late', 'absent', 'excused'] },
  note:       { type: String, default: '', maxlength: 200 },
  updated_at: { type: Date, default: Date.now },
});

markSchema.index({ class_id: 1, date: 1, student_id: 1 }, { unique: true });

module.exports = mongoose.model('AttendanceMark', markSchema);
