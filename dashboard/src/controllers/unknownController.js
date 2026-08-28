const Unknown = require('../models/Unknown');


// ============================================================
// GET ALL ACTIVE UNKNOWNS
// ============================================================

exports.getAllUnknowns = async (req, res) => {
  try {

    const unknowns =
      await Unknown.find({

        is_resolved: false,

      })
        .sort({

          last_seen: -1,

        });


    res.json({

      success: true,

      count:
        unknowns.length,

      data:
        unknowns,

    });


  } catch (err) {

    res.status(500).json({

      success: false,

      error:
        err.message,

    });

  }
};


// ============================================================
// RESOLVE UNKNOWN
// ============================================================

exports.resolveUnknown = async (req, res) => {
  try {

    const id =
      req.params.trackerId;


    // Supports:
    // new global_id
    // old tracker_id
    //
    // This keeps existing dashboard buttons working.

    const unknown =
      await Unknown.findOneAndUpdate(

        {

          $or: [

            {
              global_id: id,
            },

            {
              tracker_id: id,
            },

          ],

        },

        {

          is_resolved: true,

        },

        {

          new: true,

        }

      );


    if (!unknown) {

      return res
        .status(404)
        .json({

          success: false,

          error:
            'Unknown identity not found',

        });

    }


    res.json({

      success: true,

      data: unknown,

    });


  } catch (err) {

    res.status(500).json({

      success: false,

      error:
        err.message,

    });

  }
};


// ============================================================
// MANUAL UNKNOWN INGEST
// ============================================================

exports.ingestUnknown = async (req, res) => {
  try {

    const data =
      req.body || {};


    const globalId =

      data.global_id ||

      data.global_person_id ||

      data.tracker_id;


    if (!globalId) {

      return res
        .status(400)
        .json({

          success: false,

          error:
            'global_id is required',

        });

    }


    // --------------------------------------------------------
    // Local camera tracker
    // --------------------------------------------------------

    const localTrackerId =
      data.local_tracker_id ||
      null;


    const localTrackKey =

      localTrackerId &&
        data.camera_id

        ? `${data.camera_id}:${localTrackerId}`

        : null;


    const now =
      new Date();


    const update = {

      $set: {

        last_seen:
          now,

        last_camera:
          data.camera_id || null,

        is_resolved:
          false,

      },


      $setOnInsert: {

        global_id:
          globalId,

        // old dashboard compatibility
        tracker_id:
          globalId,

        first_seen:
          now,

      },


      $inc: {

        detection_count: 1,

      },

    };


    // --------------------------------------------------------
    // Camera history
    // --------------------------------------------------------

    if (data.camera_id) {

      update.$addToSet = {

        cameras_seen:
          data.camera_id,

      };

    }


    // --------------------------------------------------------
    // Local DeepSORT tracker history
    // --------------------------------------------------------

    if (localTrackKey) {

      update.$addToSet =
        update.$addToSet || {};


      update
        .$addToSet
        .local_tracker_ids =
        localTrackKey;

    }


    // --------------------------------------------------------
    // Snapshot
    // --------------------------------------------------------

    if (data.snapshot_path) {

      update.$set.snapshot_path =
        data.snapshot_path;

    }


    // --------------------------------------------------------
    // Save
    // --------------------------------------------------------

    const unknown =
      await Unknown.findOneAndUpdate(

        {

          global_id:
            globalId,

        },

        update,

        {

          upsert: true,

          new: true,

          setDefaultsOnInsert:
            true,

        }

      );


    res.status(201).json({

      success: true,

      data: unknown,

    });


  } catch (err) {

    console.error(
      'Unknown ingest error:',
      err
    );


    res.status(400).json({

      success: false,

      error:
        err.message,

    });

  }
};


// ============================================================
// UNKNOWN STATISTICS
// ============================================================

exports.getStats = async (req, res) => {
  try {

    const totalUnknowns =
      await Unknown.countDocuments({

        is_resolved: false,

      });


    res.json({

      success: true,

      data: {

        totalUnknowns,

      },

    });


  } catch (err) {

    res.status(500).json({

      success: false,

      error:
        err.message,

    });

  }
};