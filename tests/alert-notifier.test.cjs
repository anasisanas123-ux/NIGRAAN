const {test} = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {createNotifier, wants, readSettings} = require(path.join(__dirname, '../dashboard/src/services/alertNotifier.js'));

const quiet = {log() {}, error() {}};
const weapon = {timestamp: '2026-11-10T10:00:00Z', camera_id: 'cam_b', event_type: 'possible_weapon',
  weapon_type: 'knife', description: 'Possible knife visible (score 0.71).'};
const running = {timestamp: '2026-11-10T10:00:01Z', camera_id: 'cam_a', event_type: 'rapid_movement',
  description: 'Rapid movement cue.'};

function harness(env = {}, alerts = []) {
  const emitted = [];
  const mails = [];
  const sms = [];
  const state = {alerts};
  const notifier = createNotifier({
    env,
    logger: quiet,
    io: {emit: (name, data) => emitted.push([name, data])},
    mailer: {sendMail: async mail => mails.push(mail)},
    fetchImpl: async (url, options) => {
      sms.push(Object.fromEntries(options.body));
      return {ok: true, json: async () => ({})};
    },
    getAlerts: async () => state.alerts,
    getSnapshot: async () => ({body: Buffer.from([0xff, 0xd8]), type: 'image/jpeg'}),
  });
  return {notifier, emitted, mails, sms, state};
}

const fullEnv = {
  SMTP_HOST: 'smtp.test', ALERT_EMAIL_TO: 'a@test.com, b@test.com',
  TWILIO_ACCOUNT_SID: 'AC1', TWILIO_AUTH_TOKEN: 't', TWILIO_FROM: '+1555', ALERT_SMS_TO: '+92300',
};

test('alerts that existed before startup are not re-sent', async () => {
  const h = harness(fullEnv, [weapon]);
  assert.deepEqual(await h.notifier.poll(), []);
  assert.equal(h.emitted.length, 0);
  assert.equal(h.mails.length, 0);
});

test('a new weapon alert goes to sound, email with a snapshot, and SMS', async () => {
  const h = harness(fullEnv, []);
  await h.notifier.poll();
  h.state.alerts = [weapon];
  const fresh = await h.notifier.poll();
  assert.equal(fresh.length, 1);
  assert.equal(h.emitted[0][0], 'behavior_alert');
  assert.equal(h.emitted[0][1].sound, true);
  assert.equal(h.emitted[0][1].title, 'Possible weapon (knife)');
  assert.equal(h.mails.length, 1);
  assert.equal(h.mails[0].to, 'a@test.com, b@test.com');
  assert.match(h.mails[0].subject, /Possible weapon \(knife\) on Camera B/);
  assert.equal(h.mails[0].attachments.length, 1);
  assert.equal(h.sms.length, 1);
  assert.equal(h.sms[0].To, '+92300');
  // The same alert on the next poll is not sent again.
  await h.notifier.poll();
  assert.equal(h.mails.length, 1);
});

test('rapid movement only makes a sound by default, not email or SMS', async () => {
  const h = harness(fullEnv, []);
  await h.notifier.poll();
  h.state.alerts = [running];
  await h.notifier.poll();
  assert.equal(h.emitted.length, 1);
  assert.equal(h.mails.length, 0);
  assert.equal(h.sms.length, 0);
});

test('email and SMS are throttled per camera and alert type', async () => {
  const h = harness(fullEnv, []);
  await h.notifier.poll(0);
  h.state.alerts = [weapon];
  await h.notifier.poll(1000);
  h.state.alerts = [{...weapon, timestamp: '2026-11-10T10:00:20Z'}, weapon];
  await h.notifier.poll(21000);
  assert.equal(h.emitted.length, 2, 'every alert still reaches the dashboard');
  assert.equal(h.mails.length, 1, 'second email within 60 s is skipped');
  h.state.alerts = [{...weapon, timestamp: '2026-11-10T10:01:30Z'}];
  await h.notifier.poll(91000);
  assert.equal(h.mails.length, 2);
});

test('without settings, email and SMS stay off and the engine being down is harmless', async () => {
  assert.equal(readSettings({}).email, null);
  assert.equal(readSettings({}).sms, null);
  const notifier = createNotifier({env: {}, logger: quiet, io: null, getAlerts: async () => { throw new Error('down'); }});
  assert.deepEqual(await notifier.poll(), []);
});

test('a failed email is reported in status and does not stop SMS', async () => {
  const h = harness(fullEnv, []);
  h.notifier.stop();
  const failing = createNotifier({
    env: fullEnv, logger: quiet, io: null,
    mailer: {sendMail: async () => { throw new Error('auth failed'); }},
    fetchImpl: async () => ({ok: true, json: async () => ({})}),
  });
  const sent = await failing.sendTest();
  assert.deepEqual(sent, {sound: false, email: false, sms: true});
  assert.equal(failing.status().email.last_error, 'auth failed');
});

test('event filters', () => {
  assert.equal(wants(['all'], 'rapid_movement'), true);
  assert.equal(wants([], 'possible_weapon'), false);
  assert.equal(wants([], 'test_alert'), true);
  assert.equal(wants(['possible_weapon'], 'restricted_area_entry'), false);
});
