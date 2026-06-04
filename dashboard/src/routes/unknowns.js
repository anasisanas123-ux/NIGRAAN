const express = require('express');
const router  = express.Router();
const ctrl    = require('../controllers/unknownController');

router.get('/',                        ctrl.getAllUnknowns);
router.put('/:trackerId/resolve',      ctrl.resolveUnknown);

module.exports = router;