const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');

function setup() {
  const elements = new Map(), requests = [];
  const element = () => ({style:{},children:[],value:'',textContent:'',appendChild(v){this.children.push(v);},
    append(v){this.children.push(v);},replaceChildren(){this.children=[];},addEventListener(){},
    getContext(){return {clearRect(){},drawImage(){},strokeRect(){},fillText(){}};},
    getBoundingClientRect(){return {left:20,top:40,width:800,height:400};},setPointerCapture(){},
    showModal(){this.open=true;},close(){this.open=false;},querySelector(selector){
      if (!elements.has(selector)) elements.set(selector,element()); return elements.get(selector);
    }});
  const dialog = element();
  const context = {document:{createElement(tag){return tag==='dialog'?dialog:element();},body:{appendChild(){}},createTextNode(v){return v;}},
    window:{},Image:class {constructor(){this.naturalWidth=1920;this.naturalHeight=1080;} async decode(){}},
    URL:{createObjectURL(){return 'blob:image';},revokeObjectURL(){}},crypto:{randomUUID(){return 'test-zone';}},confirm(){return true;},
    fetch:async (url, options={}) => {
      requests.push({url,...options});
      if(url.endsWith('/snapshot')) return {ok:true,blob:async()=>({})};
      return {ok:true,json:async()=>({success:true,data:options.method==='PUT'?JSON.parse(options.body):{recognition:null,restricted:[]}})};
    }};
  vm.runInNewContext(fs.readFileSync('dashboard/public/js/zone-editor.js','utf8'),context);
  return {context,dialog,requests,el:id=>dialog.querySelector('#zone-'+id)};
}

test('drawing uses normalized displayed-image coordinates and explicit save persists settings',async()=>{
  const {context,requests,el,dialog} = setup();
  await context.window.openZoneEditor('cam_a');
  assert.equal(dialog.open,true);
  assert.equal(el('canvas').width,1920);
  el('kind').value='restricted';el('name').value='Door';el('dwell').value='12';
  el('canvas').onpointerdown({clientX:220,clientY:140,pointerId:1});
  el('canvas').onpointermove({clientX:620,clientY:340});
  el('canvas').onpointerup();
  await el('save').onclick();
  assert.equal(requests.filter(v=>v.method==='PUT').length,0,'unadded drawing cannot silently be lost on save');
  el('add').onclick(); await el('save').onclick();
  const saved = JSON.parse(requests.find(v=>v.method==='PUT').body);
  assert.deepEqual(saved.restricted[0],{id:'test-zone',name:'Door',box:[.25,.25,.75,.75],dwell_seconds:12});
  assert.equal(saved.recognition,null);
  assert.match(el('message').textContent,/Saved/);
});

test('reverse drag clamps coordinates and recognition zone can be removed',async()=>{
  const {context,requests,el} = setup();
  await context.window.openZoneEditor('cam_b');
  el('kind').value='recognition';
  el('canvas').onpointerdown({clientX:920,clientY:500,pointerId:1});
  el('canvas').onpointermove({clientX:0,clientY:0});el('canvas').onpointerup();el('add').onclick();
  await el('save').onclick();
  assert.deepEqual(JSON.parse(requests.find(v=>v.method==='PUT').body).recognition,[0,0,1,1]);
  el('list').children[0].children[1].onclick();await el('save').onclick();
  assert.equal(JSON.parse(requests.filter(v=>v.method==='PUT')[1].body).recognition,null);
});
