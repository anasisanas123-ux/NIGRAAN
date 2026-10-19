const express = require('express');
const router = express.Router();
const controller = require('../controllers/cameraController');

router.get('/health', controller.getCameraHealth);
router.get('/:cameraId/zones', controller.cameraZones);
router.put('/:cameraId/zones', controller.cameraZones);
router.get('/:cameraId/snapshot', controller.getSnapshot);
router.get('/behavior-alerts', controller.getBehaviorAlerts);

module.exports = router;
