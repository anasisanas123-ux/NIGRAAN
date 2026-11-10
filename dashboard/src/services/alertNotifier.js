// Watches the camera engine for new activity alerts (possible weapon,
// possible violence, restricted area, dwell, rapid movement) and sends them out:
//   - sound + pop-up in every open dashboard tab (socket.io "behavior_alert")
//   - email through SMTP, when SMTP_* settings are present in dashboard/.env
//   - SMS through Twilio, when TWILIO_* settings are present in dashboard/.env
// It runs inside the dashboard server, so email/SMS go out even with no tab open.
const http = require('http');

const DEFAULT_EMAIL_EVENTS = 'possible_weapon,possible_violent_activity,restricted_area_entry,expired_visitor_seen';
const DEFAULT_SMS_EVENTS = 'possible_weapon,possible_violent_activity';

const EVENT_TITLES = {
  possible_weapon: 'Possible weapon',
  possible_violent_activity: 'Possible violent activity',
  restricted_area_entry: 'Restricted area entry',
  extended_zone_dwell: 'Extended stay in zone',
  rapid_movement: 'Rapid movement',
  expired_visitor_seen: 'Expired visitor on camera',
  exam_phone: 'Exam: phone visible',
  exam_looking_around: 'Exam: looking around',
  exam_left_seat: 'Exam: left seat',
  exam_too_close: 'Exam: students too close',
};

function parseList(value, fallback = '') {
  return String(value ?? fallback).split(',').map(item => item.trim()).filter(Boolean);
}

// "all" (or "*") matches every event type; an empty list matches nothing.
// Test alerts go to every configured channel regardless of the filters.
function wants(eventTypes, eventType) {
  return eventType === 'test_alert' || eventTypes.includes('all') || eventTypes.includes('*')
    || eventTypes.includes(eventType);
}

// Engine alerts have no id, so identify them by their content.
function alertKey(alert) {
  return [alert.timestamp, alert.camera_id, alert.event_type, alert.zone_id || '',
    alert.weapon_type || '', alert.description || ''].join('|');
}

function titleFor(alert) {
  const title = EVENT_TITLES[alert.event_type] || String(alert.event_type || 'Activity alert').replace(/_/g, ' ');
  return alert.weapon_type ? `${title} (${alert.weapon_type})` : title;
}

function cameraLabel(cameraId) {
  const id = String(cameraId || 'unknown');
  const match = /^cam_([a-z0-9]+)$/i.exec(id);
  return match ? `Camera ${match[1].toUpperCase()}` : id;
}

function formatMessage(alert) {
  const when = new Date(alert.timestamp);
  const time = Number.isNaN(when.getTime()) ? String(alert.timestamp || '') : when.toLocaleString();
  return {
    title: titleFor(alert),
    camera: cameraLabel(alert.camera_id),
    time,
    text: `NIGRAAN: ${titleFor(alert)} on ${cameraLabel(alert.camera_id)} at ${time}. ${alert.description || ''}`.trim(),
  };
}

function readSettings(env = process.env) {
  const emailTo = parseList(env.ALERT_EMAIL_TO);
  const smsTo = parseList(env.ALERT_SMS_TO);
  return {
    pollMs: Math.max(1, Number(env.NOTIFY_POLL_SECONDS) || 3) * 1000,
    repeatMs: Math.max(0, Number(env.NOTIFY_REPEAT_SECONDS ?? 60) || 0) * 1000,
    soundEvents: parseList(env.NOTIFY_SOUND_EVENTS, 'all'),
    emailEvents: parseList(env.NOTIFY_EMAIL_EVENTS, DEFAULT_EMAIL_EVENTS),
    smsEvents: parseList(env.NOTIFY_SMS_EVENTS, DEFAULT_SMS_EVENTS),
    email: env.SMTP_HOST && emailTo.length ? {
      host: env.SMTP_HOST,
      port: Number(env.SMTP_PORT) || 587,
      secure: String(env.SMTP_SECURE || '').toLowerCase() === 'true',
      user: env.SMTP_USER || '',
      pass: env.SMTP_PASS || '',
      from: env.ALERT_EMAIL_FROM || env.SMTP_USER || 'nigraan@localhost',
      to: emailTo,
    } : null,
    sms: env.TWILIO_ACCOUNT_SID && env.TWILIO_AUTH_TOKEN && env.TWILIO_FROM && smsTo.length ? {
      sid: env.TWILIO_ACCOUNT_SID,
      token: env.TWILIO_AUTH_TOKEN,
      from: env.TWILIO_FROM,
      to: smsTo,
    } : null,
    enginePort: Number(env.PYTHON_ENGINE_PORT || 5001),
  };
}

function getFromEngine(port, path, { binary = false, timeout = 2500 } = {}) {
  return new Promise((resolve, reject) => {
    const request = http.get({ hostname: '127.0.0.1', port, path, timeout }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          return reject(new Error(`engine returned ${response.statusCode}`));
        }
        const body = Buffer.concat(chunks);
        if (binary) return resolve({ body, type: response.headers['content-type'] || '' });
        try { resolve(JSON.parse(body.toString('utf8'))); }
        catch (_) { reject(new Error('engine returned invalid JSON')); }
      });
      response.on('error', reject);
    });
    request.on('timeout', () => request.destroy(new Error('timeout')));
    request.on('error', reject);
  });
}

function createNotifier({ io, env = process.env, fetchImpl = globalThis.fetch, mailer = null,
  getAlerts = null, getSnapshot = null, logger = console } = {}) {
  const settings = readSettings(env);
  const seen = new Set();
  const lastSent = new Map(); // "channel|camera|event" -> ms, so a long event doesn't spam
  let primed = false;
  let timer = null;
  let transport = mailer;
  const lastError = { email: null, sms: null };

  getAlerts = getAlerts || (async () => (await getFromEngine(settings.enginePort, '/behavior-alerts')).alerts || []);
  getSnapshot = getSnapshot || (cameraId => getFromEngine(settings.enginePort,
    `/snapshot/${encodeURIComponent(cameraId)}`, { binary: true, timeout: 4000 }));

  function emailTransport() {
    if (!transport && settings.email) {
      const nodemailer = require('nodemailer');
      const { host, port, secure, user, pass } = settings.email;
      transport = nodemailer.createTransport({ host, port, secure, auth: user ? { user, pass } : undefined });
    }
    return transport;
  }

  function throttled(channel, alert, now) {
    const key = `${channel}|${alert.camera_id}|${alert.event_type}|${alert.weapon_type || ''}`;
    if (now - (lastSent.get(key) ?? -Infinity) < settings.repeatMs) return true;
    lastSent.set(key, now);
    return false;
  }

  async function sendEmail(alert, message) {
    const attachments = [];
    if (alert.camera_id && !alert.test) {
      try {
        const snapshot = await getSnapshot(alert.camera_id);
        if (snapshot.body.length && /^image\//.test(snapshot.type)) {
          attachments.push({ filename: `${alert.camera_id}.jpg`, content: snapshot.body, contentType: snapshot.type });
        }
      } catch (_) { /* send without a picture */ }
    }
    await emailTransport().sendMail({
      from: settings.email.from,
      to: settings.email.to.join(', '),
      subject: `NIGRAAN alert: ${message.title} on ${message.camera}`,
      text: `${message.title}\nCamera: ${message.camera}\nTime: ${message.time}\n\n${alert.description || ''}\n\n`
        + 'This is an automated review cue, not a confirmed event. Check the footage before acting.',
      attachments,
    });
  }

  async function sendSms(message) {
    const { sid, token, from, to } = settings.sms;
    const auth = Buffer.from(`${sid}:${token}`).toString('base64');
    await Promise.all(to.map(async number => {
      const response = await fetchImpl(`https://api.twilio.com/2010-04-01/Accounts/${encodeURIComponent(sid)}/Messages.json`, {
        method: 'POST',
        headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ From: from, To: number, Body: message.text.slice(0, 320) }),
      });
      if (!response.ok) {
        let detail = '';
        try { detail = (await response.json()).message || ''; } catch (_) { /* ignore */ }
        throw new Error(`Twilio ${response.status} for ${number}${detail ? `: ${detail}` : ''}`);
      }
    }));
  }

  async function dispatch(alert, now = Date.now()) {
    const message = formatMessage(alert);
    const result = { sound: false, email: false, sms: false };
    if (io) {
      result.sound = wants(settings.soundEvents, alert.event_type);
      io.emit('behavior_alert', { ...alert, title: message.title, camera: message.camera, sound: result.sound });
    }
    const jobs = [];
    if (settings.email && wants(settings.emailEvents, alert.event_type) && (alert.test || !throttled('email', alert, now))) {
      jobs.push(sendEmail(alert, message).then(() => { result.email = true; lastError.email = null; })
        .catch(error => { lastError.email = error.message; logger.error(`[notify] email failed: ${error.message}`); }));
    }
    if (settings.sms && wants(settings.smsEvents, alert.event_type) && (alert.test || !throttled('sms', alert, now))) {
      jobs.push(sendSms(message).then(() => { result.sms = true; lastError.sms = null; })
        .catch(error => { lastError.sms = error.message; logger.error(`[notify] SMS failed: ${error.message}`); }));
    }
    await Promise.all(jobs);
    return result;
  }

  // One poll. The first successful poll only records what already exists, so
  // starting the dashboard does not re-send old alerts.
  async function poll(now = Date.now()) {
    let alerts;
    try { alerts = await getAlerts(); }
    catch (_) { return []; } // engine not running yet; try again next poll
    const fresh = [];
    const current = new Set();
    for (const alert of alerts) {
      const key = alertKey(alert);
      current.add(key);
      if (primed && !seen.has(key)) fresh.push(alert);
    }
    // The engine only returns its latest alerts, so older keys never come back.
    seen.clear();
    for (const key of current) seen.add(key);
    primed = true;
    fresh.sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
    for (const alert of fresh) await dispatch(alert, now);
    return fresh;
  }

  function start() {
    if (timer) return;
    const loop = async () => {
      try { await poll(); } catch (error) { logger.error(`[notify] ${error.message}`); }
      timer = setTimeout(loop, settings.pollMs);
    };
    timer = setTimeout(loop, 0);
    logger.log(`Alert notifications: sound on, email ${settings.email ? 'on' : 'off'}, SMS ${settings.sms ? 'on' : 'off'}`);
  }

  function stop() { clearTimeout(timer); timer = null; }

  function status() {
    return {
      sound: { events: settings.soundEvents },
      email: { configured: Boolean(settings.email), events: settings.emailEvents,
        recipients: settings.email ? settings.email.to.length : 0, last_error: lastError.email },
      sms: { configured: Boolean(settings.sms), events: settings.smsEvents,
        recipients: settings.sms ? settings.sms.to.length : 0, last_error: lastError.sms },
      repeat_seconds: settings.repeatMs / 1000,
    };
  }

  function sendTest() {
    return dispatch({
      test: true,
      timestamp: new Date().toISOString(),
      camera_id: 'test',
      event_type: 'test_alert',
      description: 'Test notification from the NIGRAAN dashboard. No action is needed.',
    });
  }

  return { start, stop, poll, dispatch, status, sendTest };
}

module.exports = { createNotifier, alertKey, formatMessage, parseList, readSettings, wants };
