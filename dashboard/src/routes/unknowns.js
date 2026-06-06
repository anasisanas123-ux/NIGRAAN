const express = require('express');
const router  = express.Router();
const path    = require('path');
const fs      = require('fs');
const ctrl    = require('../controllers/unknownController');

router.get('/',                    ctrl.getAllUnknowns);
router.put('/:trackerId/resolve',  ctrl.resolveUnknown);

router.post('/ingest', async (req, res) => {
  try {
    const Unknown = require('../models/Unknown');
    const data    = req.body;
    await Unknown.findOneAndUpdate(
      { tracker_id: data.tracker_id },
      {
        $set:      { last_seen: new Date(), snapshot_path: data.snapshot_path },
        $inc:      { detection_count: 1 },
        $addToSet: { cameras_seen: data.camera_id }
      },
      { upsert: true, new: true }
    );
    res.status(201).json({ success: true });
  } catch(err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

router.get('/stats', async (req, res) => {
  try {
    const Unknown = require('../models/Unknown');
    const total   = await Unknown.countDocuments({ is_resolved: false });
    res.json({ success: true, data: { totalUnknowns: total } });
  } catch(err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

router.get('/snapshot/:gid', (req, res) => {
  const gid      = req.params.gid;
  const snapPath = path.join(
    __dirname, '..', '..', '..', 'data', 'snapshots', gid + '.jpg'
  );
  if(fs.existsSync(snapPath)){
    res.sendFile(snapPath);
  } else {
    res.status(404).json({ error: 'Snapshot not found' });
  }
});

module.exports = router;