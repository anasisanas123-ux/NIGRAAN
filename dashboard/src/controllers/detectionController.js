const Detection = require('../models/Detection');
const Unknown = require('../models/Unknown');

const CAMERA_NAMES = {
  cam_a: 'Camera A',
  cam_b: 'Camera B',
};


// ============================================================
// GET RECENT DETECTIONS
// ============================================================

exports.getRecentDetections = async (req, res) => {
  try {
    const limit = Math.min(
      parseInt(req.query.limit, 10) || 50,
      500
    );

    const detections = await Detection.find()
      .sort({ timestamp: -1, _id: -1 })
      .limit(limit);

    res.json({
      success: true,
      count: detections.length,
      data: detections,
    });

  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message,
    });
  }
};


// ============================================================
// GET DETECTIONS BY CAMERA
// ============================================================

exports.getDetectionsByCamera = async (req, res) => {
  try {
    const detections = await Detection.find({
      camera_id: req.params.camId,
    })
      .sort({ timestamp: -1, _id: -1 })
      .limit(100);

    res.json({
      success: true,
      count: detections.length,
      data: detections,
    });

  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message,
    });
  }
};

// Search the durable event history. All user-supplied text is escaped before
// it is used as a regular expression, and result sizes are bounded.
exports.searchDetections = async (req, res) => {
  try {
    const query = {};
    const text = String(req.query.q || '').trim().slice(0, 100);
    if (text) {
      const safe = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const pattern = new RegExp(safe, 'i');
      query.$or = [
        { person_id: pattern },
        { global_person_id: pattern },
        { person_name: pattern },
        { student_id: pattern },
        { student_name: pattern },
        { local_tracker_id: pattern },
      ];
    }
    if (req.query.camera) query.camera_id = String(req.query.camera).slice(0, 40);
    if (req.query.event) query.event_type = String(req.query.event).slice(0, 40);

    const range = {};
    if (req.query.from) {
      const from = new Date(req.query.from);
      if (Number.isNaN(from.getTime())) return res.status(400).json({ success: false, error: 'Invalid start date.' });
      range.$gte = from;
    }
    if (req.query.to) {
      const to = new Date(req.query.to);
      if (Number.isNaN(to.getTime())) return res.status(400).json({ success: false, error: 'Invalid end date.' });
      range.$lte = to;
    }
    if (range.$gte && range.$lte && range.$gte > range.$lte) {
      return res.status(400).json({ success: false, error: 'Start date must be before end date.' });
    }
    if (Object.keys(range).length) query.timestamp = range;

    const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 100, 1), 300);
    const data = await Detection.find(query).sort({ timestamp: -1, _id: -1 }).limit(limit).lean();
    res.json({ success: true, count: data.length, data });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// Return a person's camera-to-camera event sequence in chronological order.
exports.getPersonTimeline = async (req, res) => {
  try {
    const identity = String(req.params.personId || '').trim().slice(0, 120);
    if (!identity) return res.status(400).json({ success: false, error: 'A person ID is required.' });
    const data = await Detection.find({
      $or: [
        { person_id: identity },
        { global_person_id: identity },
        { student_id: identity },
      ],
    }).sort({ timestamp: 1, _id: 1 }).limit(500).lean();
    res.json({ success: true, count: data.length, data });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};


// ============================================================
// INGEST DETECTION FROM PYTHON
// ============================================================

exports.ingestDetection = async (req, res) => {
  try {
    const body = req.body || {};

    const isKnown = Boolean(body.is_known);

    const role =
      body.role ||
      (isKnown ? 'student' : 'unknown');


    // --------------------------------------------------------
    // Global identity
    // --------------------------------------------------------

    const globalPersonId =
      body.global_person_id ||
      body.person_id ||
      body.student_id ||
      (!isKnown ? body.global_id : null) ||
      null;


    const personId =
      body.person_id ||
      globalPersonId ||
      null;


    // --------------------------------------------------------
    // Camera-local DeepSORT ID
    // --------------------------------------------------------

    const localTrackerId =
      body.local_tracker_id ||
      body.tracker_id ||
      null;


    const eventType =
      body.event_type ||
      'detected';


    // --------------------------------------------------------
    // Safety duplicate check
    //
    // Python handles real event logic.
    // This prevents accidental duplicate HTTP requests.
    // --------------------------------------------------------

    if (globalPersonId) {

      const threeSecondsAgo =
        new Date(Date.now() - 3000);

      const duplicate =
        await Detection.findOne({

          global_person_id: globalPersonId,

          camera_id: body.camera_id,

          local_tracker_id: localTrackerId,

          event_type: eventType,

          timestamp: {
            $gte: threeSecondsAgo,
          },

        });


      if (duplicate) {

        return res.status(200).json({
          success: true,
          duplicate: true,
          data: duplicate,
        });

      }
    }


    // --------------------------------------------------------
    // Name
    // --------------------------------------------------------

    const personName =
      body.person_name ||
      body.student_name ||
      (
        isKnown
          ? 'Known'
          : globalPersonId || 'Unknown'
      );


    // --------------------------------------------------------
    // Save detection
    // --------------------------------------------------------

    const detection =
      await Detection.create({

        camera_id:
          body.camera_id,

        camera_name:
          CAMERA_NAMES[body.camera_id] ||
          body.camera_id,


        person_id:
          personId,

        global_person_id:
          globalPersonId,

        person_name:
          personName,


        role:
          role,

        is_known:
          isKnown,


        confidence:
          Number(body.confidence || 0),

        reid_confidence:
          Number(body.reid_confidence || 0),


        local_tracker_id:
          localTrackerId,

        // backward compatibility
        tracker_id:
          localTrackerId,


        event_type:
          eventType,

        previous_camera:
          body.previous_camera || null,


        snapshot_path:
          body.snapshot_path || null,


        student_id:
          body.student_id ||
          (isKnown ? personId : null),

        student_name:
          body.student_name ||
          personName,

      });

    // Emit as soon as the event is durable. The dashboard doesn't need to
    // wait for the separate unknown-summary upsert below.
    const io = req.app.get('io');
    if (io) {
      io.emit('new_detection', detection);
    }


    // ========================================================
    // UPDATE GLOBAL UNKNOWN RECORD
    // ========================================================

    if (!isKnown && globalPersonId) {

      const localTrackKey =
        localTrackerId
          ? `${body.camera_id}:${localTrackerId}`
          : null;


      const update = {

        $set: {

          last_seen:
            detection.timestamp,

          last_camera:
            body.camera_id,

          is_resolved:
            false,

        },


        $setOnInsert: {

          global_id:
            globalPersonId,

          // old dashboard compatibility
          tracker_id:
            globalPersonId,

          first_seen:
            detection.timestamp,

        },


        $inc: {
          detection_count: 1,
        },


        $addToSet: {
          cameras_seen: body.camera_id,
        },

      };


      if (body.snapshot_path) {

        update.$set.snapshot_path =
          body.snapshot_path;

      }


      if (localTrackKey) {

        update.$addToSet.local_tracker_ids =
          localTrackKey;

      }


      await Unknown.findOneAndUpdate(

        {
          global_id: globalPersonId,
        },

        update,

        {
          upsert: true,
          new: true,
          setDefaultsOnInsert: true,
        }

      );
    }


    res.status(201).json({
      success: true,
      data: detection,
    });


  } catch (err) {

    console.error(
      'Detection ingest error:',
      err
    );

    res.status(400).json({
      success: false,
      error: err.message,
    });

  }
};


// ============================================================
// DASHBOARD STATISTICS
// ============================================================

exports.getStats = async (req, res) => {
  try {

    const today = new Date();

    today.setHours(
      0,
      0,
      0,
      0
    );


    const [

      totalToday,

      knownToday,

      unknownToday,

      totalUnknowns,

      visitorsToday,

    ] = await Promise.all([


      Detection.countDocuments({

        timestamp: {
          $gte: today,
        },

      }),


      Detection.countDocuments({

        timestamp: {
          $gte: today,
        },

        is_known: true,

      }),


      Detection.countDocuments({

        timestamp: {
          $gte: today,
        },

        is_known: false,

      }),


      Unknown.countDocuments({

        is_resolved: false,

      }),


      Detection.countDocuments({

        timestamp: {
          $gte: today,
        },

        role: 'visitor',

      }),

    ]);


    res.json({

      success: true,

      data: {

        totalToday,

        knownToday,

        unknownToday,

        totalUnknowns,

        visitorsToday,

      },

    });


  } catch (err) {

    res.status(500).json({

      success: false,

      error: err.message,

    });

  }
};


// ============================================================
// RESET DETECTION / UNKNOWN LOGS
// ============================================================

exports.resetDetections = async (req, res) => {
  try {

    await Detection.deleteMany({});

    await Unknown.deleteMany({});


    const io =
      req.app.get('io');


    if (io) {

      io.emit('reset');

    }


    res.json({

      success: true,

      message:
        'All detection and unknown logs cleared',

    });


  } catch (err) {

    res.status(500).json({

      success: false,

      error: err.message,

    });

  }
};
