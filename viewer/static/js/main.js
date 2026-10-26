// GENESIS live viewer: renders the state streamed by viewer/server.py.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Net } from './net.js';
import { World, makeUniforms } from './world.js';
import { Sky } from './sky.js';
import { Creatures } from './creatures.js';
import { Effects } from './effects.js';
import { Post } from './post.js';
import { HUD } from './hud.js';
import { clamp, lerp, smooth, toast } from './util.js';

const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.setClearColor(0x000000);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.3, 4500);
camera.position.set(0, 80, 90);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = false;
controls.maxPolarAngle = 1.38;
controls.minDistance = 5;
controls.maxDistance = 420;
controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
controls.autoRotateSpeed = 0.45;

const U = makeUniforms();
const sky = new Sky(scene, U, 256);
const post = new Post(renderer, scene, camera);
const QUALITY = [[Math.min(devicePixelRatio, 2), 'HQ'], [Math.min(devicePixelRatio, 1.4), 'MQ'], [1, 'LQ']];
let quality = devicePixelRatio > 1.6 ? 1 : 0;

const app = {
  world: null, creatures: null, effects: null, init: null, frame: null,
  followUid: null, selectedUid: null, tool: null, director: false, overlay: 0,
};
const clockState = { tick: 0, recv: 0, tps: 0, lastChange: 0, interval: 0.1 };
let lastFrameTick = -1;

const net = new Net({
  open: () => { document.getElementById('loading-msg').textContent = 'loading world…'; },
  close: () => { document.getElementById('loading-msg').textContent = 'connection lost — retrying…'; },
  message: (m) => {
    if (m.type === 'init') onInit(m);
    else if (m.type === 'fields') app.world?.updateFields(m);
    else if (m.type === 'frame') onFrame(m);
    else if (m.type === 'detail') { if (m.uid === app.selectedUid) hud.inspector(m); }
  },
});
const hud = new HUD(net, app);

// ------------------------------------------------------------------------------- lifecycle
function onInit(init) {
  app.world?.dispose(); app.creatures?.dispose(); app.effects?.dispose();
  app.init = init;
  app.world = new World(scene, init, U);
  app.creatures = new Creatures(scene, app.world, init, U);
  app.effects = new Effects(scene, app.world, U);
  sky.size = Math.max(init.W, init.H);
  hud.init(init);
  app.select(null);
  app.followUid = null;
  lastFrameTick = -1;
  app.needCenter = true;
  resize();
  document.getElementById('loading').classList.add('done');
}

function onFrame(fr) {
  if (!app.world) return;
  const now = performance.now() / 1000;
  app.frame = fr;
  if (fr.tick !== lastFrameTick) {
    if (lastFrameTick >= 0) {
      const iv = now - clockState.lastChange;
      clockState.interval = lerp(clockState.interval, clamp(iv, 1 / 60, 0.8), 0.3);
    }
    clockState.lastChange = now;
    const dur = clamp(clockState.interval * 1.08, 1 / 45, 0.8);
    app.creatures.onFrame(fr, now, dur);
    lastFrameTick = fr.tick;
    if (app.overlay === 2 && fr.tick % 2 === 0) app.world.updateDensity(fr.data.agents, fr.data.agents.length / 8);
  }
  clockState.tick = fr.tick; clockState.recv = now; clockState.tps = fr.paused ? 0 : fr.tps;
  hud.onFrame(fr);
  for (const e of fr.fx) {
    app.effects.burst(e.k, e.x, e.y);
    directorConsider(e, now);
  }
  if (app.needCenter && app.creatures.agents.size) {
    app.needCenter = false;
    let sx = 0, sy = 0, n = 0;
    for (const s of app.creatures.agents.values()) { sx += s.tx; sy += s.ty; n++; }
    // open on the agent with the most neighbours within 12 cells
    const all = [...app.creatures.agents.values()];
    let best = all[0], bn = -1;
    for (const a of all) { let c = 0; for (const b of all) if (Math.abs(a.tx - b.tx) + Math.abs(a.ty - b.ty) < 12) c++; if (c > bn) { bn = c; best = a; } }
    app.flyTo(best.tx, best.ty, 60, true);
  }
}

// ------------------------------------------------------------------------------- camera helpers
const tmpV = new THREE.Vector3();
let fly = null;
app.flyTo = (x, y, dist = null, instant = false) => {
  const W = app.world;
  if (!W) return;
  const target = new THREE.Vector3(W.wx(x), W.heightAt(x, y), W.wz(y));
  const off = camera.position.clone().sub(controls.target);
  if (dist) off.setLength(dist);
  if (instant) { controls.target.copy(target); camera.position.copy(target).add(off); return; }
  fly = { t0: performance.now() / 1000, dur: 1.4, fromT: controls.target.clone(), toT: target, fromO: camera.position.clone().sub(controls.target), toO: off };
};
function updateFly(now) {
  if (!fly) return;
  const k = smooth(0, 1, (now - fly.t0) / fly.dur);
  controls.target.lerpVectors(fly.fromT, fly.toT, k);
  const o = tmpV.copy(fly.fromO).lerp(fly.toO, k);
  camera.position.copy(controls.target).add(o);
  if (k >= 1) fly = null;
}
app.select = (uid) => {
  app.selectedUid = uid;
  app.creatures?.select(uid);
  net.send({ cmd: 'select', uids: uid != null ? [uid] : [] });
  if (uid == null) { hud.closeInspector(); app.followUid = null; }
};
app.toggleFollow = () => {
  app.followUid = app.followUid === app.selectedUid ? null : app.selectedUid;
  if (app.followUid != null) { const s = app.creatures.get(app.followUid); if (s) app.flyTo(s.x, s.y, 28); }
  document.getElementById('insp-follow').classList.toggle('on', app.followUid != null);
};
app.setTool = (tool) => { app.tool = tool; hud.setToolButtons(tool); canvas.style.cursor = tool ? 'crosshair' : ''; };
app.setOverlay = (v) => {
  app.overlay = v; U.uOverlay.value = v;
  document.getElementById('overlay').value = String(v);
  if (v === 2) app.world?.densF.fill(0);
};
const caption = document.getElementById('director-caption');
function showCaption(txt, ms = 5000) {
  caption.textContent = txt; caption.classList.add('show');
  clearTimeout(showCaption._t); showCaption._t = setTimeout(() => caption.classList.remove('show'), ms);
}
app.toggleDirector = () => {
  app.director = !app.director;
  controls.autoRotate = app.director;
  document.getElementById('btn-director').classList.toggle('on', app.director);
  director.until = 0;
  showCaption(app.director ? '🎬 Director on: the camera cuts to notable events' : '🎬 Director off', 2500);
};

// ------------------------------------------------------------------------------- director
const director = { until: 0, prio: 0 };
const PRIO = { kill_large: 5, fire_start: 5, flood_start: 5, drought_start: 3, famine_start: 3, cold_snap_start: 3 };
function directorConsider(e, now) {
  if (!app.director) return;
  let p = PRIO[e.k] || 0;
  if (e.k === 'death') p = e.cause === 'predator' || e.cause === 'fire' ? 4 : 2;
  if (!p || (now < director.until && p <= director.prio)) return;
  director.until = now + (p >= 4 ? 10 : 7); director.prio = p;
  app.followUid = null;
  app.flyTo(e.x, e.y, p >= 5 ? 45 : 26);
  const txt = { kill_large: `🦬 Agent #${e.uid} and ${e.n - 1} others brought down a large prey`, fire_start: '🔥 A wildfire has started',
    flood_start: '🌊 A river floods its plain', drought_start: '☀ Drought: streams are drying up', famine_start: '🥀 Famine year: plants barely regrow',
    cold_snap_start: '❄ A cold snap grips the land', death: `☠ Agent #${e.uid} died of ${e.cause?.replace('_', ' ')}` }[e.k];
  showCaption(txt, 6000);
}
function directorIdle(now) {
  if (!app.director || now < director.until || !app.creatures?.agents.size) return;
  // follow an agent in the most precarious state
  let best = null, bv = 2;
  for (const s of app.creatures.agents.values()) {
    const v = Math.min(s.e, s.hy, s.w) + Math.random() * 0.15;
    if (v < bv) { bv = v; best = s; }
  }
  if (!best) return;
  director.until = now + 12; director.prio = 1;
  app.select(best.uid);
  app.followUid = best.uid;
  app.flyTo(best.x, best.y, 22);
  const need = best.hy <= Math.min(best.e, best.w) ? 'thirsty' : best.e <= best.w ? 'hungry' : 'cold';
  showCaption(`🎬 Following agent #${best.uid}, who is ${need}`, 5000);
}

// ------------------------------------------------------------------------------- input
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5 || !app.world) return;
  downAt = null;
  ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(app.world.terrainMesh, false)[0];
  const cell = hit ? app.world.cellOf(hit.point.x, hit.point.z) : null;
  if (app.tool && cell) {
    net.send({ cmd: app.tool, x: cell[0], y: cell[1] });
    toast(`${app.tool === 'fire' ? '🔥 Fire' : '🌊 Flood'} at (${cell[0]}, ${cell[1]}) (logged)`);
    app.setTool(null);
    return;
  }
  let uid = app.creatures.pick(ray);
  if (uid == null && cell) uid = app.creatures.nearest(cell[0], cell[1], 1.6);
  if (uid != null) app.select(uid);
});
const keys = new Set();
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  keys.add(e.key.toLowerCase());
  const fr = app.frame;
  if (e.key === ' ') { net.send({ cmd: fr?.paused ? 'play' : 'pause' }); e.preventDefault(); }
  else if (e.key === '.') net.send({ cmd: 'step' });
  else if (e.key >= '1' && e.key <= '6') net.send({ cmd: 'speed', value: [0.25, 1, 4, 16, 100, 0][Number(e.key) - 1] });
  else if (e.key === 'Escape') { if (app.tool) app.setTool(null); else app.select(null); }
  else if (e.key === 'f' && app.selectedUid != null) app.toggleFollow();
  else if (e.key === 'c') app.toggleDirector();
  else if (e.key === 'o') app.setOverlay((app.overlay + 1) % 4);
  else if (e.key === 'h') document.getElementById('ui').classList.toggle('hide-ui');
  else if (e.key === 'p') app.screenshot();
  else if (e.key === 'r') app.toggleRecord();
  else if (e.key === 'q' && e.shiftKey) app.cycleQuality();
  else if (e.key === '?') document.getElementById('help').classList.toggle('hidden');
});
addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
function keyboardPan(dt) {
  const d = camera.position.distanceTo(controls.target);
  const sp = d * 0.9 * dt;
  const fwd = tmpV.set(controls.target.x - camera.position.x, 0, controls.target.z - camera.position.z).normalize();
  const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
  const mv = new THREE.Vector3();
  if (keys.has('w') || keys.has('arrowup')) mv.add(fwd);
  if (keys.has('s') || keys.has('arrowdown')) mv.sub(fwd);
  if (keys.has('d') || keys.has('arrowright')) mv.add(right);
  if (keys.has('a') || keys.has('arrowleft')) mv.sub(right);
  if (mv.lengthSq()) { mv.setLength(sp); controls.target.add(mv); camera.position.add(mv); app.followUid = null; }
  const rot = (keys.has('e') ? 1 : 0) - (keys.has('q') ? 1 : 0);
  if (rot) {
    const off = camera.position.clone().sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), rot * dt * 1.4);
    camera.position.copy(controls.target).add(off);
  }
}

// ------------------------------------------------------------------------------- capture
app.screenshot = () => { app.shot = true; };
let recorder = null, chunks = [];
app.toggleRecord = () => {
  const btn = document.getElementById('btn-rec');
  if (recorder) { recorder.stop(); return; }
  const stream = canvas.captureStream(60);
  const mime = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find((m) => MediaRecorder.isTypeSupported(m));
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 24e6 });
  chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: mime });
    download(blob, `genesis_${stamp()}.${mime.includes('mp4') ? 'mp4' : 'webm'}`);
    recorder = null; btn.classList.remove('on'); toast('Clip saved');
  };
  recorder.start(250);
  btn.classList.add('on');
  toast('Recording… press R again to stop');
};
function stamp() { return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19); }
function download(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }
app.cycleQuality = () => {
  quality = (quality + 1) % QUALITY.length;
  document.getElementById('btn-quality').textContent = QUALITY[quality][1];
  resize();
  toast(`Render quality ${QUALITY[quality][1]} (pixel ratio ${QUALITY[quality][0].toFixed(2)})`);
};
document.getElementById('btn-quality').textContent = QUALITY[quality][1];

function resize() {
  const pr = QUALITY[quality][0];
  renderer.setPixelRatio(pr);
  renderer.setSize(innerWidth, innerHeight, false);
  post.setSize(innerWidth, innerHeight, pr);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  const px = innerHeight * pr / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  app.creatures?.setPx(px); app.effects?.setPx(px);
}
addEventListener('resize', resize);
resize();

// ------------------------------------------------------------------------------- frame loop
let last = performance.now() / 1000, fpsAcc = 0, fpsN = 0, fps = 60, statT = 0, miniT = 0, bytesT = 0, lastBytes = 0, kbps = 0;
const season = new THREE.Vector4();
const env = { weather: 0, temp: 10 };
let droughtV = 0, wetV = 0;
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now() / 1000;
  const dt = Math.min(0.1, now - last);
  last = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
  U.uTime.value = now % 10000;
  const init = app.init, fr = app.frame;
  if (init && fr && app.world) {
    // continuous sim clock for lighting (extrapolated between frames, never past ~1 frame)
    const tickF = clockState.tick + Math.min((now - clockState.recv) * clockState.tps, Math.max(1, clockState.tps * 0.1));
    const tod = (((tickF + init.tpd / 4) % init.tpd) / init.tpd);
    const sF = (tickF / init.season_len) % 4, sb = Math.floor(sF), blend = smooth(0.82, 1.0, sF - sb);
    season.set(0, 0, 0, 0).setComponent(sb, 1 - blend).setComponent((sb + 1) % 4, season.getComponent((sb + 1) % 4) + blend);
    U.uSeason.value.copy(season);
    env.weather = fr.weather; env.temp = fr.temp;
    droughtV = lerp(droughtV, fr.drought > 0 ? 1 : 0, 1 - Math.exp(-dt * 0.5));
    wetV = lerp(wetV, fr.weather === 1 && fr.temp >= 0 ? 1 : 0, 1 - Math.exp(-dt * (fr.weather === 1 ? 0.4 : 0.12)));
    U.uDrought.value = droughtV; U.uWet.value = wetV;
    U.uWind.value = (fr.weather === 1 ? 1.5 : 0.55) + 0.25 * Math.sin(now * 0.13);
    const dist = camera.position.distanceTo(controls.target);
    const lights = sky.update(tod, fr.weather, controls.target, dist, dt);
    keyboardPan(dt);
    // follow
    if (app.followUid != null) {
      const s = app.creatures.get(app.followUid);
      if (s && s.wpos) {
        const d = tmpV.copy(s.wpos).sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 5));
        controls.target.add(d); camera.position.add(d);
      } else if (!s) app.followUid = null;
    }
    directorIdle(now);
    updateFly(now);
    // keep the orbit target on the ground
    const [cx, cy] = app.world.cellOf(controls.target.x, controls.target.z);
    if (!fly && app.followUid == null) {
      const gy = app.world.heightAt(clamp(cx, 0, init.W - 1), clamp(cy, 0, init.H - 1));
      const dy = (gy - controls.target.y) * (1 - Math.exp(-dt * 4));
      controls.target.y += dy; camera.position.y += dy;
    }
    controls.update();
    // don't let the camera dip under the terrain
    const [kx, ky] = app.world.cellOf(camera.position.x, camera.position.z);
    if (kx >= 0 && ky >= 0 && kx < init.W && ky < init.H) {
      const minY = app.world.heightAt(kx, ky) + 2.0;
      if (camera.position.y < minY) camera.position.y = minY;
    }
    sky.follow(camera);
    app.creatures.update(now, dt, now);
    // per-agent action particles (rate-limited, only near the camera)
    if (dist < 120) for (const s of app.creatures.agents.values()) {
      if (s.action === 7 && Math.random() < dt * 3) app.effects.burst('drink', s.x, s.y);
      else if (s.action === 6 && Math.random() < dt * 2) app.effects.burst('eat', s.x, s.y);
    }
    app.effects.update(dt, now, controls.target, env);
    const tilt = clamp(1.25 - dist / 140, 0.15, 1.0);
    post.render(now, lights.night, tilt);
    if (app.shot) {
      app.shot = false;
      canvas.toBlob((b) => { download(b, `genesis_${stamp()}.png`); toast('Screenshot saved'); });
    }
    if (now - miniT > 0.25) { miniT = now; hud.drawMinimap(app.creatures, app.world, [cx, cy, dist * 0.55]); }
    if (now - statT > 0.5) {
      const db = net.bytes - lastBytes; lastBytes = net.bytes; kbps = db / 1024 / Math.max(0.1, now - bytesT); bytesT = now;
      statT = now;
      hud.engineStats({ fps, agents: app.creatures.agents.size, fauna: app.creatures.fauna.size, kbps });
    }
    window.__genesis = { fps, tick: fr.tick, agents: app.creatures.agents.size, fauna: app.creatures.fauna.size, night: lights.night,
      season: fr.season, tod: fr.tod, weather: fr.weather, temp: fr.temp, tps: fr.tps, fires: fr.fires };
  } else {
    renderer.render(scene, camera);
  }
}
animate();
window.__app = app;
app.cmd = (c) => net.send(c);   // used by the smoke/screenshot scripts
