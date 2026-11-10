const {test} = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {computeAttendance, attendanceCsv, meetsOn} = require(path.join(__dirname, '../dashboard/src/services/attendance.js'));
const {createSightingSync} = require(path.join(__dirname, '../dashboard/src/services/sightingSync.js'));

// Monday 2026-11-09, 09:00-10:30 local time, late after 10 min, counts from 08:45.
const cls = {_id: 'c1', name: 'AI Lab', camera_id: 'cam_a', days: [1, 3], start_time: '09:00', end_time: '10:30',
  late_after_minutes: 10, early_minutes: 15, student_ids: ['S1', 'S2', 'S3', 'S4']};
const students = [{student_id: 'S1', name: 'Ali'}, {student_id: 'S2', name: 'Sara'}, {student_id: 'S3', name: 'Umair'}];
const local = (h, m) => new Date(2026, 10, 9, h, m);
const seen = (id, h, m) => ({person_id: id, timestamp: local(h, m)});

test('present, late and absent after the class has ended', () => {
  const sightings = [seen('S1', 8, 50), seen('S1', 9, 30), seen('S2', 9, 25), seen('S3', 8, 30), seen('S9', 9, 5)];
  const r = computeAttendance(cls, '2026-11-09', students, sightings, [], local(11, 0));
  const status = Object.fromEntries(r.rows.map(row => [row.student_id, row.status]));
  assert.deepEqual(status, {S1: 'present', S2: 'late', S3: 'absent', S4: 'absent'});
  assert.equal(r.rows.find(x => x.student_id === 'S2').minutes_late, 25);
  assert.equal(r.rows.find(x => x.student_id === 'S4').name, '(not enrolled)');
  assert.deepEqual(r.summary, {total: 4, present: 1, late: 1, absent: 2, excused: 0, not_seen_yet: 0});
  assert.deepEqual(r.not_on_list, ['S9']);
  assert.equal(r.state, 'finished');
});

test('during the class, unseen students are "not seen yet", not absent', () => {
  const r = computeAttendance(cls, '2026-11-09', students, [seen('S1', 9, 2)], [], local(9, 20));
  assert.equal(r.state, 'in_progress');
  assert.deepEqual(r.rows.map(x => x.status), ['present', 'not_seen_yet', 'not_seen_yet', 'not_seen_yet']);
});

test('a staff correction overrides the camera and keeps the automatic status', () => {
  const marks = [{student_id: 'S3', status: 'excused', note: 'Medical'}];
  const r = computeAttendance(cls, '2026-11-09', students, [], marks, local(11, 0));
  const s3 = r.rows.find(x => x.student_id === 'S3');
  assert.equal(s3.status, 'excused');
  assert.equal(s3.automatic_status, 'absent');
  assert.equal(s3.corrected, true);
  assert.equal(r.summary.excused, 1);
});

test('class days and CSV export (formula-safe)', () => {
  assert.equal(meetsOn(cls, '2026-11-09'), true);   // Monday
  assert.equal(meetsOn(cls, '2026-11-10'), false);  // Tuesday
  const r = computeAttendance(cls, '2026-11-09', [{student_id: 'S1', name: '=HYPERLINK("x")'}], [seen('S1', 9, 0)], [], local(11, 0));
  const csv = attendanceCsv(r);
  assert.match(csv, /^Class,Date,Student ID,Name,Status/);
  assert.match(csv, /"'=HYPERLINK\(""x""\)"/);
  assert.throws(() => computeAttendance(cls, 'yesterday', [], [], []), /date/);
});

test('sightings sync stores new events once and restarts with the engine', async () => {
  const stored = [];
  const Sighting = {insertMany: async docs => {
    for (const d of docs) {
      if (stored.some(s => s.run_id === d.run_id && s.seq === d.seq)) throw Object.assign(new Error('dup'), {code: 11000});
      stored.push(d);
    }
  }};
  let engine = {run_id: 'r1', latest: 2, events: [
    {seq: 1, person_id: 'S1', person_name: 'Ali', camera_id: 'cam_a', timestamp: '2026-11-09T04:00:00Z'},
    {seq: 2, person_id: 'S2', person_name: 'Sara', camera_id: 'cam_a', timestamp: '2026-11-09T04:00:05Z'}]};
  const emitted = [];
  const sync = createSightingSync({Sighting, io: {emit: (n, d) => emitted.push(n)}, logger: {error() {}},
    fetch: async after => ({...engine, events: engine.events.filter(e => e.seq > after)})});
  assert.equal(await sync.poll(), 2);
  assert.equal(await sync.poll(), 0);
  engine = {run_id: 'r2', latest: 1, events: [{seq: 1, person_id: 'S3', camera_id: 'cam_b', timestamp: '2026-11-09T05:00:00Z'}]};
  assert.equal(await sync.poll(), 1);
  assert.deepEqual(stored.map(s => `${s.run_id}:${s.seq}:${s.person_id}`), ['r1:1:S1', 'r1:2:S2', 'r2:1:S3']);
  assert.ok(stored[0].timestamp instanceof Date);
  assert.deepEqual(emitted, ['sightings', 'sightings']);
  const down = createSightingSync({Sighting, fetch: async () => { throw new Error('ECONNREFUSED'); }});
  assert.equal(await down.poll(), 0);
});
