const express = require('express');
const router  = express.Router();
const ctrl    = require('../controllers/detectionController');

router.get('/',                  ctrl.getRecentDetections);
router.get('/stats',             ctrl.getStats);
router.get('/camera/:camId',     ctrl.getDetectionsByCamera);
router.post('/ingest',           ctrl.ingestDetection);

module.exports = router;