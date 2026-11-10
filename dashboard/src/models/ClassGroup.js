const mongoose = require('mongoose');

const TIME = /^([01]\d|2[0-3]):[0-5]\d$/;

// A class for attendance: which room camera, which weekdays and times, and
// which enrolled students belong to it.
const classSchema = new mongoose.Schema({
  name:               { type: String, required: true, trim: true, maxlength: 100 },
  camera_id:          { type: String, required: true },
  days:               { type: [Number], default: [1, 2, 3, 4, 5],
                        validate: v => v.every(d => Number.isInteger(d) && d >= 0 && d <= 6) }, // 0 = Sunday
  start_time:         { type: String, required: true, match: TIME },   // "09:00", local time
  end_time:           { type: String, required: true, match: TIME },
  late_after_minutes: { type: Number, default: 10, min: 0, max: 240 },
  // Being seen this long before the start already counts as on time.
  early_minutes:      { type: Number, default: 15, min: 0, max: 120 },
  student_ids:        { type: [String], default: [] },
  created_at:         { type: Date, default: Date.now },
});

classSchema.pre('validate', function checkTimes() {
  if (this.start_time && this.end_time && this.start_time >= this.end_time) {
    this.invalidate('end_time', 'The class must end after it starts.');
  }
});

module.exports = mongoose.model('ClassGroup', classSchema);
