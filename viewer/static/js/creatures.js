// Agents and fauna: instanced bodies, interpolation between sim ticks, procedural animation,
// need icons, selection ring, trail and perception radius.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { agentColor, skinColor, clamp, lerp } from './util.js';

const prep = (g) => { g.deleteAttribute('uv'); return g.index ? g.toNonIndexed() : g; };
const box = (w, h, d, x, y, z) => prep(new THREE.BoxGeometry(w, h, d).translate(x, y, z));
const lerpAngle = (a, b, t) => { let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; return a + d * t; };

// action ids (sim/constants.py)
const A = { NOOP: 0, REST: 5, EAT: 6, DRINK: 7, HUNT: 8, PICKUP: 9, DROP: 10 };

function iconAtlas() {
  const c = document.createElement('canvas');
  c.width = 384; c.height = 256;
  const g = c.getContext('2d');
  g.textAlign = 'center'; g.textBaseline = 'middle';
  ['🍖', '💧', '❄️', '❤️', '💤', '⚔️'].forEach((e, i) => {
    const cx = (i % 3) * 128 + 64, cy = Math.floor(i / 3) * 128 + 64;
    g.fillStyle = 'rgba(10,14,22,0.72)';
    g.beginPath(); g.arc(cx, cy, 54, 0, Math.PI * 2); g.fill();
    g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 5; g.stroke();
    g.font = '64px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif';
    g.fillText(e, cx, cy + 4);
  });
  const t = new THREE.CanvasTexture(c);
  t.flipY = false;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Creatures {
  constructor(scene, world, init, U) {
    this.scene = scene; this.world = world; this.U = U;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.cap = init.max_agents;
    this.agents = new Map();
    this.fauna = new Map();
    this.instUid = new Int32Array(this.cap);
    this.viewRadius = 5;
    // ---- agents
    const body = mergeGeometries([
      prep(new THREE.CapsuleGeometry(0.17, 0.3, 4, 10).translate(0, 0.34, 0)),
      prep(new THREE.CylinderGeometry(0.05, 0.05, 0.22, 5).translate(-0.09, 0.1, 0)),
      prep(new THREE.CylinderGeometry(0.05, 0.05, 0.22, 5).translate(0.09, 0.1, 0)),
    ]);
    const head = mergeGeometries([
      prep(new THREE.SphereGeometry(0.15, 14, 10).translate(0, 0.74, 0)),
      prep(new THREE.SphereGeometry(0.035, 6, 4).translate(-0.055, 0.76, 0.13)),
      prep(new THREE.SphereGeometry(0.035, 6, 4).translate(0.055, 0.76, 0.13)),
    ]);
    const pack = prep(new THREE.BoxGeometry(0.2, 0.2, 0.14).translate(0, 0.46, -0.2));
    const glow = (m) => {
      m.onBeforeCompile = (sh) => {
        sh.uniforms.uNight = U.uNight;
        sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uNight;')
          .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n#ifdef USE_INSTANCING_COLOR\ntotalEmissiveRadiance += vColor * (0.05 + 0.22 * uNight);\n#endif');
      };
      return m;
    };
    this.bodyMesh = new THREE.InstancedMesh(body, glow(new THREE.MeshStandardMaterial({ roughness: 0.6 })), this.cap);
    this.headMesh = new THREE.InstancedMesh(head, glow(new THREE.MeshStandardMaterial({ roughness: 0.55 })), this.cap);
    this.packMesh = new THREE.InstancedMesh(pack, new THREE.MeshStandardMaterial({ roughness: 0.7 }), this.cap);
    for (const m of [this.bodyMesh, this.headMesh, this.packMesh]) {
      m.count = 0; m.castShadow = true; m.frustumCulled = false;
      m.setColorAt(0, new THREE.Color(1, 1, 1));
      this.group.add(m);
    }
    // ---- fauna
    const rabbit = mergeGeometries([
      prep(new THREE.SphereGeometry(0.2, 10, 8).scale(0.95, 0.8, 1.25).translate(0, 0.19, 0)),
      prep(new THREE.SphereGeometry(0.11, 8, 6).translate(0, 0.3, 0.2)),
      box(0.04, 0.16, 0.05, -0.04, 0.44, 0.17), box(0.04, 0.16, 0.05, 0.04, 0.44, 0.17),
      prep(new THREE.SphereGeometry(0.06, 6, 4).translate(0, 0.22, -0.25)),
    ]);
    const bison = mergeGeometries([
      box(0.5, 0.42, 0.9, 0, 0.66, 0), box(0.54, 0.5, 0.4, 0, 0.74, 0.22),
      box(0.26, 0.28, 0.34, 0, 0.66, 0.6),
      box(0.05, 0.12, 0.05, -0.12, 0.86, 0.62), box(0.05, 0.12, 0.05, 0.12, 0.86, 0.62),
      box(0.1, 0.46, 0.1, -0.17, 0.23, 0.3), box(0.1, 0.46, 0.1, 0.17, 0.23, 0.3),
      box(0.1, 0.46, 0.1, -0.17, 0.23, -0.32), box(0.1, 0.46, 0.1, 0.17, 0.23, -0.32),
    ]);
    const wolf = mergeGeometries([
      box(0.28, 0.28, 0.72, 0, 0.47, 0), box(0.22, 0.22, 0.26, 0, 0.6, 0.45), box(0.12, 0.1, 0.16, 0, 0.55, 0.62),
      box(0.05, 0.1, 0.04, -0.07, 0.75, 0.42), box(0.05, 0.1, 0.04, 0.07, 0.75, 0.42),
      prep(new THREE.BoxGeometry(0.07, 0.07, 0.42).rotateX(0.5).translate(0, 0.5, -0.5)),
      box(0.07, 0.36, 0.07, -0.1, 0.18, 0.24), box(0.07, 0.36, 0.07, 0.1, 0.18, 0.24),
      box(0.07, 0.36, 0.07, -0.1, 0.18, -0.24), box(0.07, 0.36, 0.07, 0.1, 0.18, -0.24),
    ]);
    const fm = init.fauna_max;
    this.faunaMesh = [
      new THREE.InstancedMesh(rabbit, new THREE.MeshStandardMaterial({ roughness: 0.85 }), fm[0] + 64),
      new THREE.InstancedMesh(bison, new THREE.MeshStandardMaterial({ roughness: 0.9, flatShading: true }), fm[1] + 64),
      new THREE.InstancedMesh(wolf, new THREE.MeshStandardMaterial({ roughness: 0.75, flatShading: true }), fm[2] + 64),
    ];
    this.faunaBase = [new THREE.Color(0xcaa884), new THREE.Color(0x5e3f26), new THREE.Color(0x3b3f46)];
    for (const m of this.faunaMesh) {
      m.count = 0; m.castShadow = true; m.frustumCulled = false;
      m.setColorAt(0, new THREE.Color());
      this.group.add(m);
    }
    // predator eyes (glow at night)
    const ne = (fm[2] + 64) * 2;
    this.eyePos = new Float32Array(ne * 3);
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.BufferAttribute(this.eyePos, 3));
    this.eyes = new THREE.Points(eg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uNight: U.uNight, uPx: { value: 600 } },
      vertexShader: `uniform float uPx; void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv; gl_PointSize = uPx*0.09/-mv.z; }`,
      fragmentShader: `uniform float uNight; void main(){ float d = length(gl_PointCoord-0.5); gl_FragColor = vec4(vec3(1.0,0.25,0.1)*smoothstep(0.5,0.0,d)*(0.4+2.6*uNight), 1.0); }`,
    }));
    this.eyes.frustumCulled = false;
    this.group.add(this.eyes);
    // need icons
    const ni = this.cap;
    this.iconPos = new Float32Array(ni * 3); this.iconId = new Float32Array(ni); this.iconA = new Float32Array(ni);
    const ig = new THREE.BufferGeometry();
    ig.setAttribute('position', new THREE.BufferAttribute(this.iconPos, 3));
    ig.setAttribute('aIcon', new THREE.BufferAttribute(this.iconId, 1));
    ig.setAttribute('aAlpha', new THREE.BufferAttribute(this.iconA, 1));
    this.icons = new THREE.Points(ig, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uTex: { value: iconAtlas() }, uPx: { value: 600 } },
      vertexShader: `attribute float aIcon; attribute float aAlpha; uniform float uPx; varying float vIcon; varying float vA;
        void main(){ vIcon = aIcon; vA = aAlpha; vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv;
          gl_PointSize = clamp(uPx*0.5/-mv.z, 10.0, 46.0); }`,
      fragmentShader: `uniform sampler2D uTex; varying float vIcon; varying float vA;
        void main(){ if (vA < 0.01) discard; vec2 cr = vec2(mod(vIcon, 3.0), floor(vIcon / 3.0));
          vec4 c = texture2D(uTex, (cr + gl_PointCoord) / vec2(3.0, 2.0)); if (c.a < 0.05) discard; gl_FragColor = vec4(c.rgb, c.a*vA); }`,
    }));
    this.icons.frustumCulled = false;
    this.icons.renderOrder = 5;
    this.group.add(this.icons);
    this.showIcons = true;
    // selection
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.66, 48).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.vision = new THREE.Mesh(new THREE.RingGeometry(5.35, 5.5, 96).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0x8fd6ff, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.ring.visible = this.vision.visible = false;
    this.group.add(this.ring, this.vision);
    this.trailN = 300;
    this.trailPos = new Float32Array(this.trailN * 3);
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(this.trailPos, 3));
    tg.setDrawRange(0, 0);
    this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.trail.frustumCulled = false;
    this.group.add(this.trail);
    this.trailLen = 0; this.trailT = 0;
    this.selected = null;
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._e = new THREE.Euler(0, 0, 0, 'YXZ');
    this._p = new THREE.Vector3(); this._s = new THREE.Vector3(); this._c = new THREE.Color();
  }

  setPx(px) { this.icons.material.uniforms.uPx.value = px; this.eyes.material.uniforms.uPx.value = px; }

  onFrame(fr, now, dur) {
    const ag = fr.data.agents, n = ag.length / 8;
    const seen = this._seen = (this._seen || 0) + 1;
    for (let k = 0; k < n; k++) {
      const o = k * 8;
      const uid = ag[o + 2] | (ag[o + 3] << 16);
      const x = ag[o], y = ag[o + 1];
      let s = this.agents.get(uid);
      if (!s) {
        s = { uid, x, y, px: x, py: y, tx: x, ty: y, t0: now, dur, heading: Math.random() * 6.28, phase: Math.random() * 6, walk: 0,
          squash: 1, pitch: 0, born: now };
        const hv = ag[o + 7];
        s.hue = hv & 255; s.tone = (hv >> 8) & 15; s.size = 1.25 + ((hv >> 12) & 15) / 15 * 0.35;
        s.color = agentColor(s.hue, s.tone); s.skin = skinColor(s.tone);
        this.agents.set(uid, s);
      } else {
        s.px = s.x; s.py = s.y;
        if (Math.abs(x - s.x) + Math.abs(y - s.y) > 4) { s.px = x; s.py = y; s.x = x; s.y = y; }
      }
      s.tx = x; s.ty = y; s.t0 = now; s.dur = dur;
      const av = ag[o + 4];
      s.action = av & 255; s.carry = av >> 8;
      s.e = (ag[o + 5] & 255) / 255; s.hy = (ag[o + 5] >> 8) / 255; s.hp = (ag[o + 6] & 255) / 255; s.w = (ag[o + 6] >> 8) / 255;
      s.seen = seen;
    }
    for (const [uid, s] of this.agents) if (s.seen !== seen) this.agents.delete(uid);
    // fauna
    const fa = fr.data.fauna, nf = fa.length / 4;
    for (let k = 0; k < nf; k++) {
      const o = k * 4;
      const slot = fa[o + 3], v = fa[o + 2], sp = v & 15;
      const x = fa[o], y = fa[o + 1];
      let s = this.fauna.get(slot);
      if (!s || s.sp !== sp || Math.abs(x - s.x) + Math.abs(y - s.y) > 4) {
        s = { sp, x, y, px: x, py: y, heading: Math.random() * 6.28, phase: Math.random() * 6, walk: 0, rnd: Math.random() };
        this.fauna.set(slot, s);
      } else { s.px = s.x; s.py = s.y; }
      s.tx = x; s.ty = y; s.t0 = now; s.dur = dur;
      s.flee = (v >> 4) & 1; s.hp = ((v >> 8) & 15) / 15;
      s.seen = seen;
    }
    for (const [slot, s] of this.fauna) if (s.seen !== seen) this.fauna.delete(slot);
  }

  _interp(s, now, dt, walkGain) {
    const a = clamp((now - s.t0) / s.dur, 0, 1);
    const nx = lerp(s.px, s.tx, a), ny = lerp(s.py, s.ty, a);
    const dx = nx - s.x, dy = ny - s.y;
    const d = Math.hypot(dx, dy);
    s.x = nx; s.y = ny;
    const moving = d > 1e-4 && a < 1;
    s.walk = lerp(s.walk, moving ? 1 : 0, 1 - Math.exp(-dt * 10));
    if (d > 1e-3) s.heading = lerpAngle(s.heading, Math.atan2(dx, dy), 1 - Math.exp(-dt * 14));
    s.phase += dt * walkGain * (0.25 + s.walk);
  }

  update(now, dt, t) {
    const W = this.world;
    const m = this._m, q = this._q, e = this._e, p = this._p, sc = this._s;
    let k = 0, ki = 0;
    const CARRY_COL = [null, new THREE.Color(0xd2662e), new THREE.Color(0x7a5230), new THREE.Color(0x8a8a8a)];
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (const s of this.agents.values()) {
      this._interp(s, now, dt, 13);
      const act = s.action;
      const rest = act === A.REST ? 1 : 0;
      s.squash = lerp(s.squash, rest ? 0.62 : 1, 1 - Math.exp(-dt * 6));
      let pitch = 0, lunge = 0;
      if (act === A.EAT) pitch = 0.32 + 0.14 * Math.sin(t * 9 + s.uid);
      else if (act === A.DRINK) pitch = 0.62 + 0.06 * Math.sin(t * 5 + s.uid);
      else if (act === A.HUNT) { pitch = 0.25; lunge = 0.2 * Math.abs(Math.sin(t * 15 + s.uid)); }
      else if (act === A.PICKUP || act === A.DROP) pitch = 0.5;
      s.pitch = lerp(s.pitch, pitch, 1 - Math.exp(-dt * 10));
      const born = clamp((now - s.born) / 0.6, 0, 1);
      const bob = Math.abs(Math.sin(s.phase)) * 0.09 * s.walk;
      let sy = s.squash * (1 + 0.025 * Math.sin(t * 2.4 + s.uid)) * (1 + 0.07 * Math.sin(s.phase * 2) * s.walk);
      const sxz = 1 / Math.sqrt(sy);
      const g = W.surfaceAt(s.x, s.y);
      p.set(W.wx(s.x) + Math.sin(s.heading) * lunge, g + bob, W.wz(s.y) + Math.cos(s.heading) * lunge);
      e.set(s.pitch, s.heading, 0);
      q.setFromEuler(e);
      const z = s.size * born;
      sc.set(z * sxz, z * sy, z * sxz);
      m.compose(p, q, sc);
      this.bodyMesh.setMatrixAt(k, m); this.headMesh.setMatrixAt(k, m);
      this.bodyMesh.setColorAt(k, s.color); this.headMesh.setColorAt(k, s.skin);
      if (s.carry) { this.packMesh.setMatrixAt(k, m); this.packMesh.setColorAt(k, CARRY_COL[s.carry] || CARRY_COL[1]); }
      else this.packMesh.setMatrixAt(k, zero);
      this.instUid[k] = s.uid;
      s.wpos = s.wpos || new THREE.Vector3();
      s.wpos.set(p.x, g, p.z);
      // need icon: the most urgent body variable below 0.3, else sleeping/hunting markers
      let icon = -1, lo = 0.3;
      if (s.e < lo) { icon = 0; lo = s.e; }
      if (s.hy < lo) { icon = 1; lo = s.hy; }
      if (s.w < lo) { icon = 2; lo = s.w; }
      if (s.hp < Math.min(lo, 0.45)) icon = 3;
      if (icon < 0 && rest && W.U.uNight.value > 0.5) icon = 4;
      if (icon < 0 && act === A.HUNT) icon = 5;
      if (icon >= 0 && this.showIcons) {
        this.iconPos[ki * 3] = p.x; this.iconPos[ki * 3 + 1] = g + 1.25 * s.size + 0.06 * Math.sin(t * 3 + s.uid); this.iconPos[ki * 3 + 2] = p.z;
        this.iconId[ki] = icon; this.iconA[ki] = icon === 4 ? 0.75 : 1; ki++;
      }
      k++;
    }
    for (const mm of [this.bodyMesh, this.headMesh, this.packMesh]) {
      mm.count = k; mm.instanceMatrix.needsUpdate = true; if (mm.instanceColor) mm.instanceColor.needsUpdate = true;
    }
    const ig = this.icons.geometry;
    ig.setDrawRange(0, ki);
    ig.attributes.position.needsUpdate = ig.attributes.aIcon.needsUpdate = ig.attributes.aAlpha.needsUpdate = true;
    // fauna
    const counts = [0, 0, 0];
    let ne = 0;
    const c = this._c;
    const eyeL = new THREE.Vector3(), eyeR = new THREE.Vector3();
    for (const s of this.fauna.values()) {
      this._interp(s, now, dt, s.sp === 0 ? 16 : 9);
      const sp = s.sp, mesh = this.faunaMesh[sp];
      const j = counts[sp]++;
      if (j >= mesh.instanceMatrix.count) continue;
      const g = W.surfaceAt(s.x, s.y);
      const bob = sp === 0 ? Math.abs(Math.sin(s.phase * 0.9)) * 0.25 * s.walk : Math.abs(Math.sin(s.phase)) * 0.05 * s.walk;
      p.set(W.wx(s.x), g + bob, W.wz(s.y));
      e.set(sp === 0 ? -0.15 * s.walk * Math.sin(s.phase * 0.9) : 0, s.heading, 0);
      q.setFromEuler(e);
      const size = sp === 1 ? 1.35 + s.rnd * 0.2 : sp === 2 ? 1.25 + s.rnd * 0.15 : 1.1 + s.rnd * 0.25;
      sc.set(size, size, size);
      m.compose(p, q, sc);
      mesh.setMatrixAt(j, m);
      c.copy(this.faunaBase[sp]).multiplyScalar(0.85 + s.rnd * 0.3);
      if (sp === 1 && s.hp < 0.8) c.lerp(new THREE.Color(0x9a2015), (0.8 - s.hp) * 0.9);
      mesh.setColorAt(j, c);
      if (sp === 2 && ne < this.eyePos.length / 3 - 2) {
        eyeL.set(-0.06, 0.64, 0.58).multiplyScalar(size).applyQuaternion(q).add(p);
        eyeR.set(0.06, 0.64, 0.58).multiplyScalar(size).applyQuaternion(q).add(p);
        this.eyePos.set([eyeL.x, eyeL.y, eyeL.z, eyeR.x, eyeR.y, eyeR.z], ne * 3); ne += 2;
      }
    }
    this.faunaMesh.forEach((mesh, sp) => {
      mesh.count = Math.min(counts[sp], mesh.instanceMatrix.count);
      mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    });
    this.eyes.geometry.setDrawRange(0, ne);
    this.eyes.geometry.attributes.position.needsUpdate = true;
    // selection visuals
    const sel = this.selected != null ? this.agents.get(this.selected) : null;
    this.ring.visible = this.vision.visible = !!sel;
    if (sel) {
      const pulse = 1 + 0.08 * Math.sin(t * 5);
      this.ring.position.set(sel.wpos.x, sel.wpos.y + 0.05, sel.wpos.z);
      this.ring.scale.setScalar(pulse * sel.size);
      this.vision.position.copy(this.ring.position);
      this.vision.scale.setScalar(this.viewRadius / 5);
      if (now - this.trailT > 0.12) {
        this.trailT = now;
        const tp = this.trailPos;
        tp.copyWithin(3, 0, (this.trailN - 1) * 3);
        tp[0] = sel.wpos.x; tp[1] = sel.wpos.y + 0.12; tp[2] = sel.wpos.z;
        this.trailLen = Math.min(this.trailN, this.trailLen + 1);
        this.trail.geometry.setDrawRange(0, this.trailLen);
        this.trail.geometry.attributes.position.needsUpdate = true;
      }
    }
    this.trail.visible = !!sel;
  }

  select(uid) { if (uid !== this.selected) { this.selected = uid; this.trailLen = 0; } }
  get(uid) { return this.agents.get(uid); }
  pick(raycaster) {
    const hits = raycaster.intersectObjects([this.bodyMesh, this.headMesh], false);
    return hits.length ? this.instUid[hits[0].instanceId] : null;
  }
  nearest(x, y, maxD) {
    let best = null, bd = maxD;
    for (const s of this.agents.values()) { const d = Math.hypot(s.x - x, s.y - y); if (d < bd) { bd = d; best = s.uid; } }
    return best;
  }
  dispose() { this.scene.remove(this.group); this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); }); }
}
