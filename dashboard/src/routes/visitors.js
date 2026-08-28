const express = require("express");
const router = express.Router();
const ctrl = require("../controllers/visitorController");

router.post("/", ctrl.registerVisitor);
router.get("/", ctrl.getAllVisitors);
router.get("/active", ctrl.getActiveVisitors);
router.post("/:visitorId/checkout", ctrl.checkoutVisitor);

module.exports = router;
