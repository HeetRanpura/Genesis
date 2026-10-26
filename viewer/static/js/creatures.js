// Agents and fauna: articulated instanced bodies driven by procedural animation rigs.
// Movement: server ticks -> linear path through cells -> critically damped spring (rounded corners, smooth starts
// and stops) -> speed drives a blended idle/walk/run gait; heading turns smoothly and the body leans into turns.
// Actions pose the rig (sit to rest, kneel to drink, bend to eat, strike to hunt). Purely visual: the simulation
// itself moves agents one cell per tick.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { agentColor, skinColor, clamp, lerp } from './util.js';

const prep = (g) => { g.deleteAttribute('uv'); return g.index ? g.toNonIndexed() : g; };
const box = (w, h, d, x, y, z) => prep(new THREE.BoxGeometry(w, h, d).translate(x, y, z));
const wrap = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
const A = { NOOP: 0, REST: 5, EAT: 6, DRINK: 7, HUNT: 8, PICKUP: 9, DROP: 10 };

// critically damped spring toward a target (Game Programming Gems 4, "SmoothDamp")
function smoothDamp(cur, target, vel, smoothTime, dt) {
  const w = 2 / smoothTime, x = w * dt, e = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const ch = cur - target, tmp = (vel + w * ch) * dt;
  const v = (vel - w * tmp) * e;
  return [target + (ch + tmp) * e, v];
}

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

// ---------------------------------------------------------------------------------------- humanoid rig
// part geometry is built with its pivot (joint) at the origin, hanging along -y
const HIP_H = 0.62, THIGH = 0.3, SHIN = 0.3, UPPER = 0.27;
function humanoidParts() {
  const cap = (r, len) => prep(new THREE.CapsuleGeometry(r, len, 4, 10));
  return {
    torso: mergeGeometries([cap(0.14, 0.2).scale(1, 1, 0.75).translate(0, 0.22, 0),
      prep(new THREE.SphereGeometry(0.13, 10, 8).scale(1.05, 0.7, 0.85).translate(0, 0.02, 0))]),
    head: mergeGeometries([prep(new THREE.SphereGeometry(0.115, 14, 10).scale(0.95, 1.05, 1).translate(0, 0.11, 0)),
      prep(new THREE.SphereGeometry(0.03, 6, 4).translate(-0.045, 0.13, 0.1)),
      prep(new THREE.SphereGeometry(0.03, 6, 4).translate(0.045, 0.13, 0.1)),
      prep(new THREE.CylinderGeometry(0.045, 0.05, 0.08, 6).translate(0, -0.02, 0))]),
    thigh: cap(0.06, THIGH - 0.1).translate(0, -THIGH / 2, 0),
    shin: mergeGeometries([cap(0.05, SHIN - 0.1).translate(0, -SHIN / 2, 0),
      box(0.09, 0.05, 0.16, 0, -SHIN + 0.01, 0.04)]),
    arm: mergeGeometries([cap(0.045, UPPER - 0.06).translate(0, -UPPER / 2, 0),
      prep(new THREE.SphereGeometry(0.045, 6, 5).translate(0, -UPPER - 0.01, 0))]),
    pack: box(0.2, 0.22, 0.13, 0, 0.26, -0.15),
  };
}

// ---------------------------------------------------------------------------------------- quadruped bodies
function quadrupedParts() {
  return [
    { // small prey (hops; legs hidden in the body)
      body: mergeGeometries([
        prep(new THREE.SphereGeometry(0.2, 10, 8).scale(0.95, 0.8, 1.25).translate(0, 0.19, 0)),
        prep(new THREE.SphereGeometry(0.11, 8, 6).translate(0, 0.3, 0.2)),
        box(0.04, 0.16, 0.05, -0.04, 0.44, 0.17), box(0.04, 0.16, 0.05, 0.04, 0.44, 0.17),
        prep(new THREE.SphereGeometry(0.06, 6, 4).translate(0, 0.22, -0.25))]),
      leg: null,
    },
    { // large prey (bison)
      body: mergeGeometries([box(0.5, 0.42, 0.9, 0, 0.66, 0), box(0.54, 0.52, 0.42, 0, 0.75, 0.22),
        box(0.26, 0.28, 0.34, 0, 0.66, 0.6), box(0.05, 0.12, 0.05, -0.12, 0.86, 0.62), box(0.05, 0.12, 0.05, 0.12, 0.86, 0.62),
        prep(new THREE.BoxGeometry(0.06, 0.06, 0.3).rotateX(0.9).translate(0, 0.6, -0.5))]),
      leg: box(0.1, 0.48, 0.1, 0, -0.24, 0), hipY: 0.48, hips: [[-0.17, 0.3], [0.17, 0.3], [-0.17, -0.32], [0.17, -0.32]],
    },
    { // predator (wolf)
      body: mergeGeometries([box(0.28, 0.28, 0.72, 0, 0.47, 0), box(0.22, 0.22, 0.26, 0, 0.6, 0.45),
        box(0.12, 0.1, 0.16, 0, 0.55, 0.62), box(0.05, 0.1, 0.04, -0.07, 0.75, 0.42), box(0.05, 0.1, 0.04, 0.07, 0.75, 0.42),
        prep(new THREE.BoxGeometry(0.07, 0.07, 0.42).rotateX(0.5).translate(0, 0.5, -0.5))]),
      leg: box(0.07, 0.38, 0.07, 0, -0.19, 0), hipY: 0.38, hips: [[-0.1, 0.24], [0.1, 0.24], [-0.1, -0.24], [0.1, -0.24]],
    },
  ];
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
    const glow = (m) => {
      m.onBeforeCompile = (sh) => {
        sh.uniforms.uNight = U.uNight;
        sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uNight;')
          .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n#ifdef USE_INSTANCING_COLOR\ntotalEmissiveRadiance += vColor * (0.03 + 0.18 * uNight);\n#endif');
      };
      return m;
    };
    // ---- agents: one instanced mesh per body part (8 parts + carried bundle)
    const P = humanoidParts();
    const mk = (geo, mat) => {
      const m = new THREE.InstancedMesh(geo, mat, this.cap);
      m.count = 0; m.castShadow = true; m.receiveShadow = true; m.frustumCulled = false;
      m.setColorAt(0, new THREE.Color(1, 1, 1));
      this.group.add(m);
      return m;
    };
    const cloth = glow(new THREE.MeshStandardMaterial({ roughness: 0.78 }));
    const skin = glow(new THREE.MeshStandardMaterial({ roughness: 0.55 }));
    this.parts = {
      torso: mk(P.torso, cloth), head: mk(P.head, skin),
      thighL: mk(P.thigh, cloth), thighR: mk(P.thigh, cloth), shinL: mk(P.shin, cloth), shinR: mk(P.shin, cloth),
      armL: mk(P.arm, skin), armR: mk(P.arm, skin),
      pack: mk(P.pack, new THREE.MeshStandardMaterial({ roughness: 0.75 })),
    };
    this.pickMeshes = [this.parts.torso, this.parts.head];
    // ---- fauna
    this.Q = quadrupedParts();
    const fm = init.fauna_max;
    this.faunaBody = [];
    this.faunaLegs = [];
    const mats = [new THREE.MeshStandardMaterial({ roughness: 0.9 }), new THREE.MeshStandardMaterial({ roughness: 0.92, flatShading: true }),
      new THREE.MeshStandardMaterial({ roughness: 0.8, flatShading: true })];
    this.Q.forEach((q, sp) => {
      const n = fm[sp] + 64;
      const b = new THREE.InstancedMesh(q.body, mats[sp], n);
      b.count = 0; b.castShadow = true; b.frustumCulled = false; b.setColorAt(0, new THREE.Color());
      this.group.add(b);
      this.faunaBody.push(b);
      if (q.leg) {
        const l = new THREE.InstancedMesh(q.leg, mats[sp], n * 4);
        l.count = 0; l.castShadow = true; l.frustumCulled = false; l.setColorAt(0, new THREE.Color());
        this.group.add(l);
        this.faunaLegs.push(l);
      } else this.faunaLegs.push(null);
    });
    this.faunaBase = [new THREE.Color(0xcaa884), new THREE.Color(0x5e3f26), new THREE.Color(0x3b3f46)];
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
    // scratch objects
    this._m = new THREE.Matrix4(); this._root = new THREE.Matrix4(); this._a = new THREE.Matrix4(); this._b = new THREE.Matrix4();
    this._q = new THREE.Quaternion(); this._e = new THREE.Euler(0, 0, 0, 'YXZ'); this._p = new THREE.Vector3();
    this._s = new THREE.Vector3(); this._c = new THREE.Color();
    this._zero = new THREE.Matrix4().makeScale(0, 0, 0);
  }

  setPx(px) { this.icons.material.uniforms.uPx.value = px; this.eyes.material.uniforms.uPx.value = px; }

  // ------------------------------------------------------------------------------ network input
  onFrame(fr, now, dur) {
    const ag = fr.data.agents, n = ag.length / 8;
    const seen = this._seen = (this._seen || 0) + 1;
    for (let k = 0; k < n; k++) {
      const o = k * 8;
      const uid = ag[o + 2] | (ag[o + 3] << 16);
      const x = ag[o], y = ag[o + 1];
      let s = this.agents.get(uid);
      if (!s) {
        s = { uid, x, y, rx: x, ry: y, vx: 0, vy: 0, px: x, py: y, tx: x, ty: y, t0: now, dur, heading: Math.random() * 6.28,
          turn: 0, phase: Math.random() * 6, gait: 0, pose: {}, born: now, wpos: new THREE.Vector3() };
        const hv = ag[o + 7];
        s.hue = hv & 255; s.tone = (hv >> 8) & 15; s.size = 1.2 + ((hv >> 12) & 15) / 15 * 0.3;
        s.color = agentColor(s.hue, s.tone); s.skin = skinColor(s.tone);
        s.pants = s.color.clone().multiplyScalar(0.45).lerp(new THREE.Color(0x3a3024), 0.5);
        this.agents.set(uid, s);
      } else {
        s.px = s.x; s.py = s.y;
        if (Math.abs(x - s.x) + Math.abs(y - s.y) > 4) { s.px = s.x = s.rx = x; s.py = s.y = s.ry = y; s.vx = s.vy = 0; }
      }
      s.tx = x; s.ty = y; s.t0 = now; s.dur = dur;
      const av = ag[o + 4];
      s.action = av & 255; s.carry = av >> 8;
      s.e = (ag[o + 5] & 255) / 255; s.hy = (ag[o + 5] >> 8) / 255; s.hp = (ag[o + 6] & 255) / 255; s.w = (ag[o + 6] >> 8) / 255;
      s.seen = seen;
    }
    for (const [uid, s] of this.agents) if (s.seen !== seen) this.agents.delete(uid);
    const fa = fr.data.fauna, nf = fa.length / 4;
    for (let k = 0; k < nf; k++) {
      const o = k * 4;
      const slot = fa[o + 3], v = fa[o + 2], sp = v & 15;
      const x = fa[o], y = fa[o + 1];
      let s = this.fauna.get(slot);
      if (!s || s.sp !== sp || Math.abs(x - s.x) + Math.abs(y - s.y) > 4) {
        s = { sp, x, y, rx: x, ry: y, vx: 0, vy: 0, px: x, py: y, heading: Math.random() * 6.28, phase: Math.random() * 6,
          gait: 0, rnd: Math.random() };
        this.fauna.set(slot, s);
      } else { s.px = s.x; s.py = s.y; }
      s.tx = x; s.ty = y; s.t0 = now; s.dur = dur;
      s.flee = (v >> 4) & 1; s.hp = ((v >> 8) & 15) / 15;
      s.seen = seen;
    }
    for (const [slot, s] of this.fauna) if (s.seen !== seen) this.fauna.delete(slot);
  }

  // path (piecewise linear through cells) -> spring-smoothed render position, speed, heading, turn rate
  _move(s, now, dt, smoothT) {
    const a = clamp((now - s.t0) / s.dur, 0, 1);
    s.x = lerp(s.px, s.tx, a); s.y = lerp(s.py, s.ty, a);
    const ox = s.rx, oy = s.ry;
    [s.rx, s.vx] = smoothDamp(s.rx, s.x, s.vx, smoothT, dt);
    [s.ry, s.vy] = smoothDamp(s.ry, s.y, s.vy, smoothT, dt);
    const spd = Math.hypot(s.rx - ox, s.ry - oy) / Math.max(dt, 1e-4);
    s.speed = lerp(s.speed || 0, spd, 1 - Math.exp(-dt * 8));
    if (spd > 0.15) {
      const target = Math.atan2(s.rx - ox, s.ry - oy);
      const d = wrap(target - s.heading);
      const step = d * (1 - Math.exp(-dt * 10));
      s.heading = wrap(s.heading + step);
      s.turn = lerp(s.turn || 0, step / Math.max(dt, 1e-4), 1 - Math.exp(-dt * 6));
    } else s.turn = lerp(s.turn || 0, 0, 1 - Math.exp(-dt * 6));
  }

  // ------------------------------------------------------------------------------ per render frame
  update(now, dt, t) {
    const W = this.world, P = this.parts;
    const root = this._root, A1 = this._a, B1 = this._b, M = this._m, q = this._q, e = this._e, p = this._p, sc = this._s;
    const night = this.U.uNight.value;
    let k = 0, ki = 0;
    const CARRY_COL = [null, new THREE.Color(0xd2662e), new THREE.Color(0x7a5230), new THREE.Color(0x8a8a8a)];
    const setPart = (mesh, mat, col) => { mesh.setMatrixAt(k, mat); if (col) mesh.setColorAt(k, col); };
    for (const s of this.agents.values()) {
      this._move(s, now, dt, 0.11);
      const act = s.action;
      // gait blend: 0 idle .. 1 walk .. 2 run (cells per second)
      const g = clamp(s.speed / 3.0, 0, 2.2);
      s.gait = lerp(s.gait, g, 1 - Math.exp(-dt * 6));
      const gw = Math.min(1, s.gait), gr = clamp(s.gait - 1, 0, 1);
      s.phase += dt * (2.0 + 5.5 * gw + 4.0 * gr) * (s.gait > 0.05 ? 1 : 0.2);
      // target pose from the current action (blended over ~0.25 s)
      const pose = s.pose;
      const tgt = { sit: 0, kneel: 0, bend: 0, reach: 0, strike: 0 };
      if (act === A.REST && s.gait < 0.3) tgt.sit = 1;
      else if (act === A.DRINK) tgt.kneel = 1;
      else if (act === A.EAT || act === A.PICKUP || act === A.DROP) { tgt.bend = 1; tgt.reach = 1; }
      else if (act === A.HUNT) tgt.strike = 1;
      for (const key in tgt) pose[key] = lerp(pose[key] || 0, tgt[key], 1 - Math.exp(-dt * 7));
      const ph = s.phase, amp = 0.55 * gw + 0.25 * gr;
      const legL = Math.sin(ph) * amp, legR = Math.sin(ph + Math.PI) * amp;
      const kneeOf = (lp) => (0.12 + 0.9 * Math.max(0, Math.sin(lp - 0.6))) * (gw + 0.6 * gr);
      const breathe = 0.012 * Math.sin(t * 2.2 + s.uid);
      // pose blending
      const sit = pose.sit, kneel = pose.kneel, bend = pose.bend, strike = pose.strike;
      let thL = legL * (1 - sit - kneel) - 1.45 * sit - 0.25 * kneel, thR = legR * (1 - sit - kneel) - 1.45 * sit - 1.25 * kneel;
      let knL = kneeOf(ph) * (1 - sit - kneel) + 1.5 * sit + 1.55 * kneel, knR = kneeOf(ph + Math.PI) * (1 - sit - kneel) + 1.5 * sit + 1.4 * kneel;
      const hipDrop = 0.33 * sit + 0.24 * kneel + 0.08 * bend + Math.abs(Math.sin(ph * 2)) * -0.035 * (gw + gr);
      const torsoPitch = 0.08 * gw + 0.18 * gr + 0.85 * bend + 0.75 * kneel - 0.05 * sit + 0.25 * strike + breathe;
      const swing = 0.7 * gw + 0.4 * gr;
      let arL = -legL * swing / Math.max(amp, 0.01) * amp * 0.9 - 0.9 * pose.reach - 0.3 * kneel + 0.15 * sit;
      let arR = -legR * swing / Math.max(amp, 0.01) * amp * 0.9 - 0.9 * pose.reach - 0.3 * kneel + 0.15 * sit
        - strike * (0.6 + 1.2 * Math.abs(Math.sin(t * 11 + s.uid)));
      const roll = clamp(-s.turn * 0.06, -0.25, 0.25) * (gw + gr);
      // root transform
      const z = s.size * clamp((now - s.born) / 0.6, 0.01, 1);
      const ground = W.surfaceAt(s.rx, s.ry);
      p.set(W.wx(s.rx), ground, W.wz(s.ry));
      e.set(0, s.heading, roll);
      q.setFromEuler(e);
      root.compose(p, q, sc.set(z, z, z));
      s.wpos.set(p.x, ground, p.z);
      const hipY = HIP_H - hipDrop;
      // torso: pivot at the hips, pitched forward
      A1.makeTranslation(0, hipY, 0).premultiply(root);
      e.set(torsoPitch, 0, 0); B1.makeRotationFromEuler(e);
      const torsoM = A1.clone().multiply(B1);
      setPart(P.torso, torsoM, s.color);
      // head: on top of the torso, nods while eating, looks up slightly when running
      e.set(-0.15 * gr + 0.35 * bend * (0.5 + 0.5 * Math.sin(t * 9 + s.uid)) - 0.2 * kneel, 0, 0);
      M.makeTranslation(0, 0.42, 0).multiply(B1.makeRotationFromEuler(e));
      setPart(P.head, torsoM.clone().multiply(M), s.skin);
      // arms from the shoulders
      for (const [mesh, side, ang] of [[P.armL, -1, arL], [P.armR, 1, arR]]) {
        e.set(ang, 0, side * (0.08 + 0.1 * gr));
        M.makeTranslation(side * 0.19, 0.36, 0).multiply(B1.makeRotationFromEuler(e));
        setPart(mesh, torsoM.clone().multiply(M), s.skin);
      }
      // legs from the hips: thigh, then shin at the knee
      for (const [thM, shM, side, th, kn] of [[P.thighL, P.shinL, -1, thL, knL], [P.thighR, P.shinR, 1, thR, knR]]) {
        e.set(th, 0, 0);
        const hip = A1.clone().multiply(M.makeTranslation(side * 0.09, 0, 0)).multiply(B1.makeRotationFromEuler(e));
        setPart(thM, hip, s.pants);
        e.set(kn, 0, 0);
        setPart(shM, hip.multiply(M.makeTranslation(0, -THIGH, 0)).multiply(B1.makeRotationFromEuler(e)), s.pants);
      }
      if (s.carry) setPart(P.pack, torsoM, CARRY_COL[s.carry] || CARRY_COL[1]);
      else P.pack.setMatrixAt(k, this._zero);
      this.instUid[k] = s.uid;
      // need icon: the most urgent body variable below 0.3, else sleeping/hunting markers
      let icon = -1, lo = 0.3;
      if (s.e < lo) { icon = 0; lo = s.e; }
      if (s.hy < lo) { icon = 1; lo = s.hy; }
      if (s.w < lo) { icon = 2; lo = s.w; }
      if (s.hp < Math.min(lo, 0.45)) icon = 3;
      if (icon < 0 && act === A.REST && night > 0.5) icon = 4;
      if (icon < 0 && act === A.HUNT) icon = 5;
      if (icon >= 0 && this.showIcons) {
        this.iconPos[ki * 3] = p.x; this.iconPos[ki * 3 + 1] = ground + (1.35 - 0.35 * sit) * s.size + 0.05 * Math.sin(t * 3 + s.uid);
        this.iconPos[ki * 3 + 2] = p.z; this.iconId[ki] = icon; this.iconA[ki] = icon === 4 ? 0.75 : 1; ki++;
      }
      k++;
    }
    for (const mm of Object.values(P)) {
      mm.count = k; mm.instanceMatrix.needsUpdate = true; if (mm.instanceColor) mm.instanceColor.needsUpdate = true;
    }
    const ig = this.icons.geometry;
    ig.setDrawRange(0, ki);
    ig.attributes.position.needsUpdate = ig.attributes.aIcon.needsUpdate = ig.attributes.aAlpha.needsUpdate = true;
    this._updateFauna(now, dt, t);
    this._updateSelection(now, t);
  }

  _updateFauna(now, dt, t) {
    const W = this.world, root = this._root, M = this._m, B1 = this._b, q = this._q, e = this._e, p = this._p, sc = this._s, c = this._c;
    const counts = [0, 0, 0], legN = [0, 0, 0];
    let ne = 0;
    const eye = new THREE.Vector3();
    for (const s of this.fauna.values()) {
      this._move(s, now, dt, s.sp === 0 ? 0.07 : 0.14);
      const sp = s.sp, body = this.faunaBody[sp], Q = this.Q[sp];
      const j = counts[sp];
      if (j >= body.instanceMatrix.count) continue;
      counts[sp]++;
      const g = clamp(s.speed / 2.5, 0, 2);
      s.gait = lerp(s.gait, g, 1 - Math.exp(-dt * 6));
      s.phase += dt * (sp === 0 ? 4 + 9 * s.gait : 2 + 6 * s.gait);
      const ground = W.surfaceAt(s.rx, s.ry);
      const size = sp === 1 ? 1.35 + s.rnd * 0.2 : sp === 2 ? 1.25 + s.rnd * 0.15 : 1.1 + s.rnd * 0.25;
      const graze = sp !== 2 && s.gait < 0.15 ? 0.5 + 0.5 * Math.sin(t * 0.7 + s.rnd * 20) : 0;
      let bob, pitch;
      if (sp === 0) { const h = Math.max(0, Math.sin(s.phase)); bob = h * 0.28 * Math.min(1, s.gait); pitch = -0.25 * Math.cos(s.phase) * Math.min(1, s.gait); }
      else { bob = Math.abs(Math.sin(s.phase * 2)) * 0.035 * s.gait; pitch = 0.18 * graze; }
      p.set(W.wx(s.rx), ground + bob, W.wz(s.ry));
      e.set(pitch, s.heading, clamp(-(s.turn || 0) * 0.04, -0.2, 0.2));
      q.setFromEuler(e);
      root.compose(p, q, sc.set(size, size, size));
      body.setMatrixAt(j, root);
      c.copy(this.faunaBase[sp]).multiplyScalar(0.85 + s.rnd * 0.3);
      if (sp === 1 && s.hp < 0.8) c.lerp(new THREE.Color(0x9a2015), (0.8 - s.hp) * 0.9);
      body.setColorAt(j, c);
      const legs = this.faunaLegs[sp];
      if (legs) {
        const amp = 0.55 * Math.min(1, s.gait) + 0.25 * Math.max(0, s.gait - 1);
        Q.hips.forEach(([hx, hz], li) => {
          const diag = (li === 0 || li === 3) ? 0 : Math.PI;          // trot: diagonal pairs move together
          e.set(Math.sin(s.phase + diag) * amp, 0, 0);
          M.makeTranslation(hx, Q.hipY, hz).multiply(B1.makeRotationFromEuler(e));
          legs.setMatrixAt(legN[sp], root.clone().multiply(M));
          legs.setColorAt(legN[sp], c);
          legN[sp]++;
        });
      }
      if (sp === 2 && ne < this.eyePos.length / 3 - 2) {
        for (const ex of [-0.06, 0.06]) {
          eye.set(ex, 0.64, 0.58).applyMatrix4(root);
          this.eyePos.set([eye.x, eye.y, eye.z], ne * 3); ne++;
        }
      }
    }
    this.faunaBody.forEach((m, sp) => {
      m.count = counts[sp]; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true;
      const l = this.faunaLegs[sp];
      if (l) { l.count = legN[sp]; l.instanceMatrix.needsUpdate = true; if (l.instanceColor) l.instanceColor.needsUpdate = true; }
    });
    this.eyes.geometry.setDrawRange(0, ne);
    this.eyes.geometry.attributes.position.needsUpdate = true;
  }

  _updateSelection(now, t) {
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
    const hits = raycaster.intersectObjects(this.pickMeshes, false);
    return hits.length ? this.instUid[hits[0].instanceId] : null;
  }
  nearest(x, y, maxD) {
    let best = null, bd = maxD;
    for (const s of this.agents.values()) { const d = Math.hypot(s.x - x, s.y - y); if (d < bd) { bd = d; best = s.uid; } }
    return best;
  }
  dispose() { this.scene.remove(this.group); this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); }); }
}
