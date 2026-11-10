// Attendance rules, kept free of the database so they can be tested directly.
//
// For one class on one date:
//   present  first seen on the class camera between (start - early_minutes)
//            and (start + late_after_minutes)
//   late     first seen after that, before the class ends
//   absent   not seen on the class camera during the class
// A staff correction (AttendanceMark) overrides the automatic status.

function parseDate(date) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(date || ''));
  if (!match) throw new Error('Use a date like 2026-11-10.');
  return [Number(match[1]), Number(match[2]) - 1, Number(match[3])];
}

function at(date, time, extraMinutes = 0) {
  const [y, m, d] = parseDate(date);
  const [hh, mm] = String(time).split(':').map(Number);
  return new Date(y, m, d, hh, mm + extraMinutes, 0, 0); // server local time
}

function localDateString(when = new Date()) {
  const pad = n => String(n).padStart(2, '0');
  return `${when.getFullYear()}-${pad(when.getMonth() + 1)}-${pad(when.getDate())}`;
}

function classWindow(cls, date) {
  return {
    from: at(date, cls.start_time, -(cls.early_minutes ?? 15)),
    start: at(date, cls.start_time),
    lateAfter: at(date, cls.start_time, cls.late_after_minutes ?? 10),
    end: at(date, cls.end_time),
  };
}

function meetsOn(cls, date) {
  const [y, m, d] = parseDate(date);
  return (cls.days || []).includes(new Date(y, m, d).getDay());
}

/**
 * cls: class document; students: [{student_id, name}]; sightings: [{person_id,
 * timestamp}] on the class camera; marks: [{student_id, status, note}].
 * now: current time, so a class still in progress shows "not yet seen".
 */
function computeAttendance(cls, date, students, sightings, marks = [], now = new Date()) {
  const w = classWindow(cls, date);
  const firstSeen = new Map();
  const lastSeen = new Map();
  for (const s of sightings) {
    const t = new Date(s.timestamp);
    if (t < w.from || t > w.end) continue;
    const id = String(s.person_id);
    if (!firstSeen.has(id) || t < firstSeen.get(id)) firstSeen.set(id, t);
    if (!lastSeen.has(id) || t > lastSeen.get(id)) lastSeen.set(id, t);
  }
  const marksById = new Map(marks.map(m => [String(m.student_id), m]));
  const names = new Map(students.map(s => [String(s.student_id), s.name]));
  const inProgress = now >= w.from && now <= w.end;
  const notStarted = now < w.from;

  const rows = (cls.student_ids || []).map(id => {
    const first = firstSeen.get(String(id)) || null;
    let status;
    if (first) status = first <= w.lateAfter ? 'present' : 'late';
    else if (notStarted) status = 'upcoming';
    else if (inProgress) status = 'not_seen_yet';
    else status = 'absent';
    const mark = marksById.get(String(id));
    return {
      student_id: String(id),
      name: names.get(String(id)) || '(not enrolled)',
      status: mark ? mark.status : status,
      automatic_status: status,
      corrected: Boolean(mark),
      note: mark ? mark.note || '' : '',
      first_seen: first ? first.toISOString() : null,
      last_seen: lastSeen.get(String(id))?.toISOString() || null,
      minutes_late: first && first > w.start ? Math.round((first - w.start) / 60000) : 0,
    };
  });
  const count = status => rows.filter(r => r.status === status).length;
  // Recognised on the class camera but not on the class list.
  const rosterSet = new Set((cls.student_ids || []).map(String));
  const visitors = [...firstSeen.keys()].filter(id => !rosterSet.has(id));
  return {
    class_id: String(cls._id || ''),
    class_name: cls.name,
    camera_id: cls.camera_id,
    date,
    meets_today: meetsOn(cls, date),
    window: { from: w.from.toISOString(), start: w.start.toISOString(), late_after: w.lateAfter.toISOString(), end: w.end.toISOString() },
    state: notStarted ? 'upcoming' : inProgress ? 'in_progress' : 'finished',
    summary: { total: rows.length, present: count('present'), late: count('late'), absent: count('absent'),
      excused: count('excused'), not_seen_yet: count('not_seen_yet') },
    rows,
    not_on_list: visitors,
  };
}

function csvCell(value) {
  const text = String(value ?? '');
  // Prefix formula-like cells so spreadsheets do not execute them.
  const safe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

function attendanceCsv(report) {
  const header = ['Class', 'Date', 'Student ID', 'Name', 'Status', 'First seen', 'Minutes late', 'Corrected', 'Note'];
  const lines = [header.join(',')];
  for (const r of report.rows) {
    lines.push([report.class_name, report.date, r.student_id, r.name, r.status,
      r.first_seen ? new Date(r.first_seen).toLocaleTimeString() : '', r.minutes_late,
      r.corrected ? 'yes' : '', r.note].map(csvCell).join(','));
  }
  return lines.join('\r\n') + '\r\n';
}

module.exports = { computeAttendance, attendanceCsv, localDateString, meetsOn, classWindow, csvCell };
