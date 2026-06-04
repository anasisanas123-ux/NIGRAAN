const express = require('express');
const router  = express.Router();
const ctrl    = require('../controllers/detectionController');
const Detection = require('../models/Detection');
const Unknown   = require('../models/Unknown');

router.get('/',                  ctrl.getRecentDetections);
router.get('/stats',             ctrl.getStats);
router.get('/camera/:camId',     ctrl.getDetectionsByCamera);
router.post('/ingest',           ctrl.ingestDetection);

router.delete('/reset', async (req, res) => {
  try {
    await Detection.deleteMany({});
    await Unknown.deleteMany({});
    res.json({ success: true, message: 'All logs cleared' });
  } catch(err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;