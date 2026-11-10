const express = require('express');
const ClassGroup = require('../models/ClassGroup');
const AttendanceMark = require('../models/AttendanceMark');
const Sighting = require('../models/Sighting');
const Student = require('../models/Student');
const { computeAttendance, attendanceCsv, localDateString, classWindow, meetsOn } = require('../services/attendance');

const router = express.Router();
const fail = (res, error) => res.status(error.statusCode || (error.name === 'ValidationError' || error.name === 'CastError' ? 400 : 500))
  .json({ success: false, error: error.message });

function classFields(body) {
  const ids = Array.isArray(body.student_ids) ? body.student_ids : [];
  return {
    name: String(body.name || '').trim(),
    camera_id: String(body.camera_id || ''),
    days: Array.isArray(body.days) ? body.days.map(Number) : undefined,
    start_time: body.start_time,
    end_time: body.end_time,
    late_after_minutes: body.late_after_minutes === undefined ? undefined : Number(body.late_after_minutes),
    early_minutes: body.early_minutes === undefined ? undefined : Number(body.early_minutes),
    student_ids: [...new Set(ids.map(String).filter(id => /^[A-Za-z0-9_-]{1,64}$/.test(id)))],
  };
}

async function report(cls, date) {
  const w = classWindow(cls, date);
  const [students, sightings, marks] = await Promise.all([
    Student.find({ student_id: { $in: cls.student_ids } }).select('student_id name').lean(),
    Sighting.find({ camera_id: cls.camera_id, timestamp: { $gte: w.from, $lte: w.end } }).select('person_id timestamp').lean(),
    AttendanceMark.find({ class_id: String(cls._id), date }).lean(),
  ]);
  return computeAttendance(cls, date, students, sightings, marks);
}

router.get('/classes', async (req, res) => {
  try { res.json({ success: true, data: await ClassGroup.find().sort({ name: 1 }).lean() }); } catch (e) { fail(res, e); }
});

router.post('/classes', async (req, res) => {
  try {
    const fields = Object.fromEntries(Object.entries(classFields(req.body || {})).filter(([, v]) => v !== undefined));
    res.status(201).json({ success: true, data: await ClassGroup.create(fields) });
  } catch (e) { fail(res, e); }
});

router.put('/classes/:id', async (req, res) => {
  try {
    const cls = await ClassGroup.findById(req.params.id);
    if (!cls) return res.status(404).json({ success: false, error: 'Class not found.' });
    for (const [key, value] of Object.entries(classFields(req.body || {}))) if (value !== undefined) cls[key] = value;
    await cls.save();
    res.json({ success: true, data: cls });
  } catch (e) { fail(res, e); }
});

router.delete('/classes/:id', async (req, res) => {
  try {
    const cls = await ClassGroup.findByIdAndDelete(req.params.id);
    if (!cls) return res.status(404).json({ success: false, error: 'Class not found.' });
    await AttendanceMark.deleteMany({ class_id: String(cls._id) });
    res.json({ success: true });
  } catch (e) { fail(res, e); }
});

// Classes meeting today, with live counts (for the "now" view).
router.get('/today', async (req, res) => {
  try {
    const date = localDateString();
    const classes = (await ClassGroup.find().lean()).filter(cls => meetsOn(cls, date));
    const data = await Promise.all(classes.map(cls => report(cls, date)));
    data.sort((a, b) => a.window.start.localeCompare(b.window.start));
    res.json({ success: true, date, data });
  } catch (e) { fail(res, e); }
});

router.get('/classes/:id/report', async (req, res) => {
  try {
    const cls = await ClassGroup.findById(req.params.id).lean();
    if (!cls) return res.status(404).json({ success: false, error: 'Class not found.' });
    const result = await report(cls, req.query.date || localDateString());
    if (req.query.format === 'csv') {
      const safeName = cls.name.replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 40) || 'class';
      res.set('Content-Type', 'text/csv; charset=utf-8');
      res.set('Content-Disposition', `attachment; filename="attendance_${safeName}_${result.date}.csv"`);
      return res.send('﻿' + attendanceCsv(result));
    }
    res.json({ success: true, data: result });
  } catch (e) { fail(res, /date/i.test(e.message) ? Object.assign(e, { statusCode: 400 }) : e); }
});

router.put('/classes/:id/marks', async (req, res) => {
  try {
    const { date, student_id, status, note } = req.body || {};
    const cls = await ClassGroup.findById(req.params.id).lean();
    if (!cls) return res.status(404).json({ success: false, error: 'Class not found.' });
    if (!cls.student_ids.includes(String(student_id))) return res.status(400).json({ success: false, error: 'That student is not in this class.' });
    if (!status || status === 'auto') {
      await AttendanceMark.deleteOne({ class_id: String(cls._id), date, student_id: String(student_id) });
    } else {
      await AttendanceMark.findOneAndUpdate(
        { class_id: String(cls._id), date, student_id: String(student_id) },
        { status, note: String(note || '').slice(0, 200), updated_at: new Date() },
        { upsert: true, runValidators: true, setDefaultsOnInsert: true },
      );
    }
    res.json({ success: true, data: await report(cls, date) });
  } catch (e) { fail(res, e); }
});

// Sightings for one person (used by the visitor/student history views).
router.get('/sightings/:personId', async (req, res) => {
  try {
    const data = await Sighting.find({ person_id: req.params.personId }).sort({ timestamp: -1 }).limit(200).lean();
    res.json({ success: true, data });
  } catch (e) { fail(res, e); }
});

module.exports = router;
