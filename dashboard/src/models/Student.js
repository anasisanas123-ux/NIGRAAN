const mongoose = require('mongoose');

const studentSchema = new mongoose.Schema({
  student_id: { type: String, required: true, unique: true },
  name:       { type: String, required: true },
  department: { type: String, default: 'Unknown' },
  embedding_path: { type: String },
  photo_path:     { type: String },
  registered_at:  { type: Date, default: Date.now }
});

module.exports = mongoose.model('Student', studentSchema);