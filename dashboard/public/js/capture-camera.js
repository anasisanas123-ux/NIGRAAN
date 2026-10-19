/* Share the engine's open camera, or explicitly use a separate browser webcam. */
class CaptureCamera {
  constructor(video, select) {
    this.video = video;
    this.select = select;
    this.image = document.createElement('img');
    this.image.alt = 'Live enrollment camera';
    this.image.hidden = true;
    video.parentNode.appendChild(this.image);
    this.generation = 0;
  }

  stop() {
    this.generation++;
    clearTimeout(this.timer);
    this.cancelDecode?.();
    this.controller?.abort();
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.image.hidden = true;
    this.ready = false;
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
    this.image.removeAttribute('src');
  }

  async start(onError) {
    this.stop();
    const generation = this.generation;
    const source = this.select.value;
    if (source === 'browser') {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Browser webcam requires localhost or HTTPS. Select an engine camera, or upload photos.');
      }
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({audio:false, video:{width:{ideal:640}, height:{ideal:480}}});
      } catch (error) {
        throw new Error(`Browser webcam: ${error.message}. If Python is already using it, select Camera A instead.`);
      }
      if (generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return false; }
      this.stream = stream;
      this.video.hidden = false;
      this.video.srcObject = stream;
      await this.video.play();
      await new Promise((resolve, reject) => {
        if (this.video.videoWidth) return resolve();
        const timer = setTimeout(() => { cleanup(); reject(new Error('No webcam frame received.')); }, 5000);
        const loaded = () => { cleanup(); resolve(); };
        const cleanup = () => { clearTimeout(timer); this.cancelDecode = null; this.video.removeEventListener('loadeddata', loaded); };
        this.cancelDecode = () => { cleanup(); resolve(); };
        this.video.addEventListener('loadeddata', loaded);
      });
      if (generation !== this.generation) return false;
      this.ready = true;
      return true;
    }
    this.video.hidden = true;
    this.image.hidden = false;
    const update = async () => {
      this.controller = new AbortController();
      const controller = this.controller;
      const timeout = setTimeout(() => controller.abort(), 7000);
      let response;
      try {
        response = await fetch(`/api/cameras/${encodeURIComponent(source)}/snapshot`, {cache:'no-store', signal:controller.signal});
      } catch (error) {
        if (generation !== this.generation) return false;
        throw new Error(error.name === 'AbortError' ? 'Camera preview timed out. Check the Python engine.' : error.message);
      } finally { clearTimeout(timeout); }
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Camera is offline.');
      }
      const blob = await response.blob();
      if (generation !== this.generation) return false;
      const url = URL.createObjectURL(blob);
      const previous = this.url;
      this.url = url;
      await new Promise((resolve, reject) => {
        const cleanup = () => {
          clearTimeout(timer);
          this.cancelDecode = null;
          this.image.onload = this.image.onerror = null;
        };
        const timer = setTimeout(() => { cleanup(); reject(new Error('No camera image received.')); }, 5000);
        this.cancelDecode = () => { cleanup(); resolve(); };
        this.image.onload = () => { cleanup(); resolve(); };
        this.image.onerror = () => { cleanup(); reject(new Error('Camera image could not be decoded.')); };
        this.image.src = url;
      });
      if (previous) URL.revokeObjectURL(previous);
      if (generation !== this.generation) return false;
      this.ready = true;
      this.timer = setTimeout(() => update().catch(error => {
        if (generation !== this.generation) return;
        this.stop();
        onError(error);
      }), 250);
      return true;
    };
    return update();
  }

  source() {
    if (!this.ready) throw new Error('Start the camera and wait for its preview first.');
    return this.stream ? this.video : this.image;
  }
}
