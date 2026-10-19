const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function controller({engineError, record = null} = {}) {
  const calls = [];
  const exports = {};
  const student = {findOneAndDelete: async query => {calls.push(['database', query.student_id]);return record;}};
  const engine = {removeIdentity: async id => {
    calls.push(['engine', id]);
    if (engineError) throw new Error(engineError);
  }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../dashboard/src/controllers/studentController.js'), 'utf8'), {
    exports, __dirname:path.join(__dirname, '../dashboard/src/controllers'), console,
    require(name) {
      if (name === '../models/Student') return student;
      if (name === '../utils/pythonEngine') return engine;
      if (name === 'fs') return {promises:{readdir:async () => []}};
      if (name === 'path') return path;
      throw new Error('Unexpected dependency');
    },
  });
  return {remove:exports.deleteStudent, calls};
}

function response() {
  return {code:200, status(code) {this.code=code;return this;}, json(body) {this.body=body;return this;}};
}

test('missing database record still removes an obsolete live enrollment', async () => {
  const {remove,calls} = controller();
  const res = response();
  await remove({params:{id:'S002'}},res);
  assert.equal(res.code,200);
  assert.equal(res.body.success,true);
  assert.deepEqual(calls,[['engine','S002'],['database','S002']]);
});

test('engine failure preserves the database record and reports failure', async () => {
  const {remove,calls} = controller({engineError:'offline'});
  const res = response();
  await remove({params:{id:'S002'}},res);
  assert.equal(res.code,503);
  assert.equal(res.body.success,false);
  assert.deepEqual(calls,[['engine','S002']]);
});

test('invalid identity never reaches either store', async () => {
  const {remove,calls} = controller();
  const res = response();
  await remove({params:{id:'../S002'}},res);
  assert.equal(res.code,400);
  assert.deepEqual(calls,[]);
});
