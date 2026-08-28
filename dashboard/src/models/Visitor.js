const mongoose = require("mongoose");

const visitorSchema = new mongoose.Schema({
  visitor_id:     { type: String, required: true, unique: true }, // V001, V002...
  name:           { type: String, required: true },
  contact:        { type: String, default: null },
  purpose:        { type: String, default: null },
  host_name:      { type: String, default: null },  // who they're visiting
  host_id:        { type: String, default: null },  // student/faculty id if applicable
  photo_path:     { type: String, default: null },  // captured photo on disk
  embedding_path: { type: String, default: null },  // .npy file path

  valid_from:     { type: Date, default: Date.now },
  valid_until:    { type: Date, required: true },

  approved_by:    { type: String, default: null },
  is_active:      { type: Boolean, default: true },  // false once expired/checked out
  checked_in_at:  { type: Date, default: Date.now },
  checked_out_at: { type: Date, default: null },

  created_at:     { type: Date, default: Date.now },
});

visitorSchema.index({ is_active: 1, valid_until: 1 });

module.exports = mongoose.model("Visitor", visitorSchema);
