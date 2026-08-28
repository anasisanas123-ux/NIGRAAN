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
      .sort({ timestamp: -1 })
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
      .sort({ timestamp: -1 })
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


    // ========================================================
    // SOCKET.IO LIVE DASHBOARD UPDATE
    // ========================================================

    const io =
      req.app.get('io');

    if (io) {

      io.emit(
        'new_detection',
        detection
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