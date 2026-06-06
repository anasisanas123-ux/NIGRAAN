const express = require("express");
const router = express.Router();
const ctrl = require("../controllers/detectionController");

router.get("/", ctrl.getRecentDetections);
router.get("/stats", ctrl.getStats);
router.get("/:camId", ctrl.getDetectionsByCamera);
router.post("/ingest", ctrl.ingestDetection);
router.delete("/reset", ctrl.resetDetections);

module.exports = router;
