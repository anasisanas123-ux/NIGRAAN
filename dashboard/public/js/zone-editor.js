/* Draw in source-image coordinates so responsive layouts do not shift zones. */
(() => {
  const dialog = document.createElement('dialog');
  dialog.style.cssText = 'width:min(1000px,95vw);max-height:95vh;overflow:auto;background:#141d2b;color:#eee;border:1px solid #667;padding:20px';
  dialog.innerHTML = `<h2 id="zone-heading">Camera zones</h2>
    <p>Drag a rectangle on the camera image. Restricted zones create entry and dwell review alerts. A recognition zone limits face recognition and supplies occupancy metrics; leave it unset for the full image.</p>
    <label>Zone type <select id="zone-kind"><option value="restricted">Restricted area</option><option value="recognition">Recognition / occupancy area</option></select></label>
    <label>Name <input id="zone-name" maxlength="100" value="Restricted area"></label>
    <label>Dwell seconds <input id="zone-dwell" type="number" min="0" max="86400" value="10"></label>
    <div><canvas id="zone-canvas" style="width:100%;height:auto;touch-action:none;cursor:crosshair"></canvas></div>
    <p id="zone-message" role="status" aria-live="polite"></p>
    <div id="zone-list"></div>
    <button type="button" id="zone-add">Add drawn zone</button>
    <button type="button" id="zone-refresh">Refresh camera image</button>
    <button type="button" id="zone-save">Save zones</button>
    <button type="button" id="zone-close">Close</button>`;
  document.body.appendChild(dialog);
  const el = id => dialog.querySelector(`#zone-${id}`);
  const canvas = el('canvas'), ctx = canvas.getContext('2d');
  let camera, settings, image, start, drawn, loading = false, dirty = false;
  async function api(method, value) {
    const response = await fetch(`/api/cameras/${encodeURIComponent(camera)}/zones`, {
      method, cache:'no-store', headers:{'Content-Type':'application/json'},
      ...(value ? {body:JSON.stringify(value)} : {}),
    });
    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.error || 'Could not load zones.');
    return result.data;
  }
  function paint() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if (image) ctx.drawImage(image,0,0);
    const box = (values, color, label) => {
      if (!values) return;
      const [a,b,c,d] = values;
      ctx.strokeStyle = color; ctx.lineWidth = Math.max(2,canvas.width/400);
      ctx.strokeRect(a*canvas.width,b*canvas.height,(c-a)*canvas.width,(d-b)*canvas.height);
      ctx.fillStyle = color; ctx.font = `${Math.max(14,canvas.width/65)}px sans-serif`;
      ctx.fillText(label,a*canvas.width+4,Math.max(18,b*canvas.height+20));
    };
    box(settings?.recognition, '#65d8ff', 'Recognition');
    for (const zone of settings?.restricted || []) box(zone.box,'#ffb257',zone.name);
    box(drawn, '#70ff91', 'Drawn selection');
  }
  function list() {
    el('list').replaceChildren();
    const row = (text, remove) => {
      const item = document.createElement('p'), button = document.createElement('button');
      item.append(document.createTextNode(text+' ')); button.textContent = 'Remove';
      button.onclick = () => {remove(); dirty = true; list(); paint();};
      item.appendChild(button); el('list').appendChild(item);
    };
    if (settings.recognition) row('Recognition / occupancy area', () => settings.recognition = null);
    settings.restricted.forEach((zone,i) => row(`${zone.name} — dwell ${zone.dwell_seconds}s`, () => settings.restricted.splice(i,1)));
  }
  async function snapshot() {
    const response = await fetch(`/api/cameras/${encodeURIComponent(camera)}/snapshot`, {cache:'no-store'});
    if (!response.ok) throw new Error('No camera image. Start this camera and try again.');
    const url = URL.createObjectURL(await response.blob());
    try {
      const next = new Image(); next.src = url; await next.decode(); image = next;
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; paint();
    } finally {URL.revokeObjectURL(url);}
  }
  function position(event) {
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)), Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
  }
  canvas.onpointerdown = event => {
    if (loading || !image) return;
    start = position(event); canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove = event => {
    if (!start) return;
    const end = position(event);
    drawn = [Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])]; paint();
  };
  canvas.onpointerup = () => {start = null;};
  canvas.onpointercancel = () => {start = null; drawn = null; paint();};
  el('add').onclick = () => {
    if (!settings || loading) return;
    if (!drawn || drawn[2]-drawn[0]<.005 || drawn[3]-drawn[1]<.005) {el('message').textContent = 'Draw a larger rectangle first.'; return;}
    if (el('kind').value === 'recognition') settings.recognition = drawn;
    else {
      const name = el('name').value.trim(), dwell = Number(el('dwell').value);
      if (!name || !Number.isFinite(dwell) || dwell<0 || dwell>86400 || settings.restricted.length>=20) {
        el('message').textContent = 'Enter a name and dwell time from 0 to 86400. Maximum 20 restricted zones.'; return;
      }
      settings.restricted.push({id:crypto.randomUUID(),name,box:drawn,dwell_seconds:dwell});
    }
    drawn = null; dirty = true; list(); paint(); el('message').textContent = 'Zone added locally. Click Save zones to apply.';
  };
  el('save').onclick = async () => {
    if (loading || !settings) return;
    if (drawn) {el('message').textContent = 'Click Add drawn zone before saving.'; return;}
    loading = true;
    try {settings = await api('PUT',settings); dirty = false; el('message').textContent = 'Saved. Zones are active and will survive an engine restart.';}
    catch(error) {el('message').textContent = error.message;}
    finally {loading = false;}
  };
  el('refresh').onclick = async () => {
    if (loading) return;
    loading = true;
    try {await snapshot();} catch(error) {el('message').textContent = error.message;} finally {loading = false;}
  };
  function close(event) {
    if (loading || ((dirty || drawn) && !confirm('Discard unsaved zone changes?'))) {event?.preventDefault(); return;}
    dialog.close();
  }
  el('close').onclick = close;
  dialog.addEventListener('cancel', close);
  window.openZoneEditor = async id => {
    if (dialog.open) return;
    camera = id; settings = null; image = null; drawn = null; dirty = false; loading = true;
    el('list').replaceChildren(); paint();
    el('heading').textContent = `${id} — camera zones`; el('message').textContent = 'Loading camera and zones…';
    dialog.showModal();
    try {settings = await api('GET'); list(); await snapshot(); el('message').textContent = 'Draw on this snapshot, add the zone, then save.';}
    catch(error) {el('message').textContent = error.message;}
    finally {loading = false;}
  };
})();
