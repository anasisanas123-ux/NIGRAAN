const Unknown = require('../models/Unknown');

exports.getAllUnknowns = async (req, res) => {
  try {
    const unknowns = await Unknown.find({ is_resolved: false })
      .sort({ last_seen: -1 });
    res.json({ success: true, count: unknowns.length, data: unknowns });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.resolveUnknown = async (req, res) => {
  try {
    const unknown = await Unknown.findOneAndUpdate(
      { tracker_id: req.params.trackerId },
      { is_resolved: true },
      { new: true }
    );
    if (!unknown) return res.status(404).json({ success: false, error: 'Tracker not found' });
    res.json({ success: true, data: unknown });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};