const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function setup({fetch, getUserMedia} = {}) {
  let image;
  const context = vm.createContext({
    document: {createElement() {
      image = {hidden:true, naturalWidth:640, naturalHeight:480,
        removeAttribute() {}, set src(value) { queueMicrotask(() => this.onload?.()); }};
      return image;
    }},
    navigator: {mediaDevices: getUserMedia ? {getUserMedia} : undefined},
    fetch: fetch || (async () => ({ok:true, blob:async () => new Blob(['frame'])})),
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
    AbortController, setTimeout, clearTimeout,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/public/js/capture-camera.js'), 'utf8')+
    '\nglobalThis.Camera = CaptureCamera;', context);
  const video = {parentNode:{appendChild() {}}, srcObject:null, hidden:false};
  const select = {value:'cam_a'};
  const camera = new context.Camera(video, select);
  return {camera, video, select, image};
}

test('engine capture works without browser mediaDevices and releases preview on stop', async () => {
  let requested;
  const {camera, image} = setup({fetch:async url => {
    requested=url;return {ok:true, blob:async () => new Blob(['frame'])};
  }});
  try {
    assert.equal(await camera.start(assert.fail), true);
    assert.equal(requested, '/api/cameras/cam_a/snapshot');
    assert.equal(camera.source(), image);
  } finally {camera.stop();}
  assert.equal(image.hidden, true);
  assert.throws(() => camera.source(), /Start the camera/);
});

test('offline camera surfaces the engine error instead of enabling capture', async () => {
  const {camera} = setup({fetch:async () => ({ok:false, json:async () => ({error:'Camera is offline.'})})});
  try {
    await assert.rejects(camera.start(assert.fail), /Camera is offline/);
    assert.throws(() => camera.source(), /Start the camera/);
  } finally {camera.stop();}
});

test('stopping during a pending snapshot cannot restore a stale preview', async () => {
  let complete;
  const {camera} = setup({fetch:() => new Promise(resolve => {complete=resolve;})});
  const pending = camera.start(assert.fail);
  camera.stop();
  complete({ok:true, blob:async () => new Blob(['frame'])});
  assert.equal(await pending, false);
  assert.equal(camera.ready, false);
});

test('browser webcam reports secure-context and camera-in-use errors', async () => {
  const missing = setup();missing.select.value='browser';
  await assert.rejects(missing.camera.start(assert.fail), /localhost or HTTPS/);
  const busy = setup({getUserMedia:async () => {throw new Error('Device in use');}});
  busy.select.value='browser';
  await assert.rejects(busy.camera.start(assert.fail), /select Camera A/);
});

test('browser webcam tracks are stopped when capture closes', async () => {
  let stopped=0;
  const stream={getTracks:() => [{stop:() => stopped++}]};
  const {camera, video, select} = setup({getUserMedia:async () => stream});
  select.value='browser';video.videoWidth=640;video.play=async () => {};
  assert.equal(await camera.start(assert.fail), true);
  assert.equal(camera.source(), video);
  camera.stop();
  assert.equal(stopped, 1);
});

test('dashboard initializes capture controls and landing page together', () => {
  const elements = new Map();
  const element = () => ({style:{}, hidden:false, value:'cam_a',
    parentNode:{appendChild() {}}, addEventListener() {}, removeAttribute() {},
    classList:{add() {}, remove() {}, contains() {return false;}}});
  const context = vm.createContext({
    document:{createElement:element, getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    }},
    setInterval:() => 1, clearInterval() {}, clearTimeout() {},
    URL:{revokeObjectURL() {}},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/public/js/capture-camera.js'), 'utf8'), context);
  const html = fs.readFileSync(path.join(__dirname, '../dashboard/public/index.html'), 'utf8');
  vm.runInContext(html.split('<script>').pop().split('</script>')[0], context);
  assert.equal(elements.get('page-dashboard').style.display, 'none');
  assert.equal(elements.get('e-capture').disabled, true);
  assert.equal(elements.get('visitor-capture-btn').disabled, true);
});
