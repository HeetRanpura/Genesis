// Particles and weather: rain, snow, fireflies, falling leaves/petals, fire (flames, smoke, light),
// and event bursts (deaths, kills, births, co-feeding, signals) with expanding rings.
import * as THREE from 'three';
import { clamp, lerp, T } from './util.js';

const SPRITE_FRAG = /* glsl */`
varying vec3 vCol; varying float vA;
void main(){ float d = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.05, d) * vA; if (a < 0.003) discard; gl_FragColor = vec4(vCol * a, a); }`;

class Pool {   // CPU-simulated particles
  constructor(scene, n, additive) {
    this.n = n; this.i = 0;
    this.pos = new Float32Array(n * 3); this.col = new Float32Array(n * 3); this.size = new Float32Array(n); this.alpha = new Float32Array(n);
    this.vel = new Float32Array(n * 3); this.life = new Float32Array(n); this.max = new Float32Array(n); this.grav = new Float32Array(n);
    this.drag = new Float32Array(n); this.size0 = new Float32Array(n); this.grow = new Float32Array(n); this.a0 = new Float32Array(n);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute('aCol', new THREE.BufferAttribute(this.col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));
    g.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1));
    this.mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.CustomBlending,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      uniforms: { uPx: { value: 600 } },
      vertexShader: `attribute vec3 aCol; attribute float aSize; attribute float aAlpha; uniform float uPx; varying vec3 vCol; varying float vA;
        void main(){ vCol = aCol; vA = aAlpha; vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv; gl_PointSize = min(uPx*aSize/-mv.z, 160.0); }`,
      fragmentShader: SPRITE_FRAG,
    });
    this.points = new THREE.Points(g, this.mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 6;
    scene.add(this.points);
  }
  emit(x, y, z, vx, vy, vz, life, size, r, g, b, opts = {}) {
    const i = this.i; this.i = (this.i + 1) % this.n;
    this.pos.set([x, y, z], i * 3); this.vel.set([vx, vy, vz], i * 3); this.col.set([r, g, b], i * 3);
    this.life[i] = life; this.max[i] = life; this.size0[i] = size; this.grav[i] = opts.grav ?? 0; this.drag[i] = opts.drag ?? 0.5;
    this.grow[i] = opts.grow ?? 0; this.a0[i] = opts.alpha ?? 1;
  }
  update(dt) {
    const { n, pos, vel, life, max, alpha, size } = this;
    for (let i = 0; i < n; i++) {
      if (life[i] <= 0) { alpha[i] = 0; continue; }
      life[i] -= dt;
      const t = 1 - life[i] / max[i];
      const k = Math.exp(-this.drag[i] * dt);
      vel[i * 3] *= k; vel[i * 3 + 1] = vel[i * 3 + 1] * k - this.grav[i] * dt; vel[i * 3 + 2] *= k;
      pos[i * 3] += vel[i * 3] * dt; pos[i * 3 + 1] += vel[i * 3 + 1] * dt; pos[i * 3 + 2] += vel[i * 3 + 2] * dt;
      alpha[i] = this.a0[i] * Math.min(1, t * 8) * (1 - t) * (life[i] > 0 ? 1 : 0);
      size[i] = this.size0[i] * (1 + this.grow[i] * t);
    }
    const g = this.points.geometry;
    g.attributes.position.needsUpdate = g.attributes.aCol.needsUpdate = g.attributes.aSize.needsUpdate = g.attributes.aAlpha.needsUpdate = true;
  }
}

function wrapBoxMaterial(opts) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: opts.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    uniforms: Object.assign({ uCenter: { value: new THREE.Vector3() }, uBox: { value: new THREE.Vector3(...opts.box) }, uTime: { value: 0 },
      uAmount: { value: 0 }, uPx: { value: 600 }, uWind: { value: new THREE.Vector2(0.3, 0.1) } }, opts.uniforms || {}),
    vertexShader: opts.vert, fragmentShader: opts.frag,
  });
}
const WRAP = /* glsl */`
uniform vec3 uCenter; uniform vec3 uBox; uniform float uTime; uniform float uAmount; uniform float uPx; uniform vec2 uWind;
vec3 wrapPos(vec3 b, float fall) {
  vec3 p;
  p.xz = uCenter.xz + (fract(b.xz - uCenter.xz / uBox.xz + uWind * uTime / uBox.xz) - 0.5) * uBox.xz;
  p.y = uCenter.y - uBox.y * 0.25 + fract(b.y - uTime * fall / uBox.y) * uBox.y;
  return p;
}`;

export class Effects {
  constructor(scene, world, U) {
    this.scene = scene; this.world = world; this.U = U;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.glow = new Pool(this.group, 5000, true);
    this.soft = new Pool(this.group, 3000, false);
    // rain (GPU streaks)
    const nr = 9000;
    const rb = new Float32Array(nr * 2 * 4), re = new Float32Array(nr * 2);
    for (let i = 0; i < nr; i++) {
      const b = [Math.random(), Math.random(), Math.random(), Math.random()];
      rb.set(b, i * 8); rb.set(b, i * 8 + 4); re[i * 2 + 1] = 1;
    }
    const rg = new THREE.BufferGeometry();
    rg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nr * 2 * 3), 3));
    rg.setAttribute('aBase', new THREE.BufferAttribute(rb, 4));
    rg.setAttribute('aEnd', new THREE.BufferAttribute(re, 1));
    this.rain = new THREE.LineSegments(rg, wrapBoxMaterial({
      box: [140, 45, 140],
      vert: `${WRAP} attribute vec4 aBase; attribute float aEnd; varying float vA;
        void main(){ vec3 p = wrapPos(aBase.xyz, 34.0); p.xz -= vec2(0.25, 0.08) * aEnd * 0.9; p.y += aEnd * 0.9;
          vA = step(aBase.w, uAmount) * 0.36; gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0); }`,
      frag: `varying float vA; void main(){ if (vA < 0.01) discard; gl_FragColor = vec4(0.75, 0.82, 0.95, vA); }`,
    }));
    this.rain.frustumCulled = false;
    this.group.add(this.rain);
    // snow
    const ns = 7000, sb = new Float32Array(ns * 4);
    for (let i = 0; i < ns * 4; i++) sb[i] = Math.random();
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ns * 3), 3));
    sg.setAttribute('aBase', new THREE.BufferAttribute(sb, 4));
    this.snow = new THREE.Points(sg, wrapBoxMaterial({
      box: [130, 40, 130],
      vert: `${WRAP} attribute vec4 aBase; varying float vA;
        void main(){ vec3 p = wrapPos(aBase.xyz, 2.2); p.x += sin(uTime * 0.9 + aBase.w * 30.0) * 0.6; p.z += cos(uTime * 0.7 + aBase.w * 17.0) * 0.4;
          vA = step(aBase.w, uAmount); vec4 mv = viewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = uPx * (0.07 + aBase.w * 0.06) / -mv.z; }`,
      frag: `varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.1, d) * vA * 0.9; if (a < 0.01) discard; gl_FragColor = vec4(vec3(0.95, 0.97, 1.0), a); }`,
    }));
    this.snow.frustumCulled = false;
    this.group.add(this.snow);
    // falling leaves (autumn) / petals (spring)
    const nl = 1600, lb = new Float32Array(nl * 4);
    for (let i = 0; i < nl * 4; i++) lb[i] = Math.random();
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nl * 3), 3));
    lg.setAttribute('aBase', new THREE.BufferAttribute(lb, 4));
    this.leaves = new THREE.Points(lg, wrapBoxMaterial({
      box: [110, 22, 110], uniforms: { uColA: { value: new THREE.Color() }, uColB: { value: new THREE.Color() } },
      vert: `${WRAP} attribute vec4 aBase; varying float vA; varying float vT; varying float vRot;
        void main(){ vec3 p = wrapPos(aBase.xyz, 1.1); p.x += sin(uTime * 1.3 + aBase.w * 40.0) * 1.2; vT = aBase.w; vRot = uTime * (1.0 + aBase.w * 3.0) + aBase.w * 9.0;
          vA = step(aBase.w, uAmount); vec4 mv = viewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = uPx * 0.16 / -mv.z; }`,
      frag: `uniform vec3 uColA; uniform vec3 uColB; varying float vA; varying float vT; varying float vRot;
        void main(){ vec2 c = gl_PointCoord - 0.5; float s = sin(vRot), co = cos(vRot); c = mat2(co, -s, s, co) * c; c.y *= 2.2;
          float a = smoothstep(0.42, 0.3, length(c)) * vA; if (a < 0.02) discard; gl_FragColor = vec4(mix(uColA, uColB, vT), a); }`,
    }));
    this.leaves.frustumCulled = false;
    this.group.add(this.leaves);
    // fireflies: CPU-placed near the camera target over vegetation, animated on the GPU
    const nfly = 700;
    this.flyBase = new Float32Array(nfly * 3);
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(this.flyBase, 3));
    const fr = new Float32Array(nfly);
    for (let i = 0; i < nfly; i++) fr[i] = Math.random();
    fg.setAttribute('aR', new THREE.BufferAttribute(fr, 1));
    this.flies = new THREE.Points(fg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: U.uTime, uAmount: { value: 0 }, uPx: { value: 600 } },
      vertexShader: `attribute float aR; uniform float uTime; uniform float uAmount; uniform float uPx; varying float vA;
        void main(){ vec3 p = position + vec3(sin(uTime*0.6 + aR*40.)*1.3, 0.6 + sin(uTime*0.9 + aR*20.)*0.5, cos(uTime*0.5 + aR*30.)*1.3);
          vA = uAmount * step(aR, uAmount) * (0.35 + 0.65 * pow(0.5 + 0.5*sin(uTime*3.0 + aR*60.), 3.0));
          vec4 mv = modelViewMatrix*vec4(p,1.); gl_Position = projectionMatrix*mv; gl_PointSize = uPx*0.12/-mv.z; }`,
      fragmentShader: `varying float vA; void main(){ float d = length(gl_PointCoord-0.5); float a = smoothstep(0.5, 0.0, d); gl_FragColor = vec4(vec3(0.75,1.0,0.35)*a*vA*2.2, 1.0); }`,
    }));
    this.flies.frustumCulled = false;
    this.group.add(this.flies);
    this.flyCenter = new THREE.Vector3(1e9, 0, 0);
    // fire lights
    this.fireLights = [0, 1, 2].map(() => { const l = new THREE.PointLight(0xff7a2a, 0, 26, 1.6); this.group.add(l); return l; });
    // expanding rings
    this.rings = [];
    for (let i = 0; i < 40; i++) {
      const r = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.0, 48).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
      r.visible = false; r.userData = { life: 0 };
      this.group.add(r); this.rings.push(r);
    }
    this.ringI = 0;
    this.rainAmt = 0; this.snowAmt = 0; this.leafAmt = 0; this.flyAmt = 0;
  }

  setPx(px) {
    for (const o of [this.snow, this.leaves, this.flies, this.rain]) o.material.uniforms.uPx.value = px;
    this.glow.mat.uniforms.uPx.value = px; this.soft.mat.uniforms.uPx.value = px;
  }

  ring(x, y, color, maxR = 4, life = 1.4) {
    const r = this.rings[this.ringI]; this.ringI = (this.ringI + 1) % this.rings.length;
    const W = this.world;
    r.position.set(W.wx(x), W.surfaceAt(x, y) + 0.15, W.wz(y));
    r.material.color.set(color);
    r.userData = { life, max: life, maxR };
    r.visible = true;
  }

  // world-cell coordinates
  burst(kind, x, y) {
    const W = this.world, X = W.wx(x), Z = W.wz(y), Y = W.surfaceAt(x, y);
    const R = Math.random;
    const g = this.glow;
    switch (kind) {
      case 'death':
        g.emit(X, Y + 0.8, Z, 0, 0.9, 0, 3.2, 0.9, 0.7, 0.85, 1.0, { drag: 0.2, alpha: 0.9 });
        for (let i = 0; i < 18; i++) g.emit(X, Y + 0.6, Z, (R() - 0.5) * 1.2, 0.6 + R(), (R() - 0.5) * 1.2, 1.8 + R(), 0.18, 0.6, 0.75, 1.0, { drag: 0.8 });
        this.ring(x, y, 0x9fc4ff, 2.5, 1.6);
        break;
      case 'birth':
        for (let i = 0; i < 26; i++) g.emit(X + (R() - 0.5) * 0.6, Y + R() * 0.3, Z + (R() - 0.5) * 0.6, 0, 1.2 + R() * 1.6, 0, 1.4 + R(), 0.16, 1.0, 0.82, 0.4, { drag: 0.6 });
        this.ring(x, y, 0xffd27a, 1.8, 1.0);
        break;
      case 'kill_small':
        for (let i = 0; i < 16; i++) g.emit(X, Y + 0.3, Z, (R() - 0.5) * 3, R() * 2.5, (R() - 0.5) * 3, 0.6 + R() * 0.4, 0.14, 1.0, 0.18, 0.1, { grav: 5 });
        break;
      case 'kill_large':
        for (let i = 0; i < 60; i++) g.emit(X, Y + 0.6, Z, (R() - 0.5) * 5, R() * 4, (R() - 0.5) * 5, 0.9 + R() * 0.7, 0.2, 1.0, 0.55 + R() * 0.3, 0.15, { grav: 4 });
        this.ring(x, y, 0xffb347, 6, 1.8);
        break;
      case 'attack_large':
        for (let i = 0; i < 6; i++) g.emit(X, Y + 0.6, Z, (R() - 0.5) * 2, R() * 1.5, (R() - 0.5) * 2, 0.4, 0.12, 1.0, 0.45, 0.15, { grav: 3 });
        break;
      case 'predator_attack':
        for (let i = 0; i < 12; i++) g.emit(X, Y + 0.5, Z, (R() - 0.5) * 2.5, R() * 2, (R() - 0.5) * 2.5, 0.5, 0.16, 1.0, 0.05, 0.12, { grav: 4 });
        this.ring(x, y, 0xff3355, 1.6, 0.6);
        break;
      case 'scrounge':
        for (let i = 0; i < 5; i++) g.emit(X, Y + 0.9, Z, (R() - 0.5) * 0.6, 0.6 + R() * 0.5, (R() - 0.5) * 0.6, 0.9, 0.12, 1.0, 0.85, 0.3, { drag: 1 });
        break;
      case 'signal':
        this.ring(x, y, 0x7fd4ff, 9, 1.2);
        break;
      case 'fire_start':
        this.ring(x, y, 0xff6a2a, 8, 1.6);
        break;
      case 'flood_start':
        this.ring(x, y, 0x4aa8ff, 30, 3.0);
        break;
      case 'drink':
        for (let i = 0; i < 3; i++) this.soft.emit(X + (R() - 0.5) * 0.5, Y + 0.15, Z + (R() - 0.5) * 0.5, (R() - 0.5) * 0.8, 1.2 + R(), (R() - 0.5) * 0.8, 0.5, 0.09, 0.55, 0.8, 1.0, { grav: 6 });
        break;
      case 'eat':
        g.emit(X + (R() - 0.5) * 0.4, Y + 0.3, Z + (R() - 0.5) * 0.4, (R() - 0.5) * 0.5, 0.8, (R() - 0.5) * 0.5, 0.7, 0.1, 0.5, 1.0, 0.3, { drag: 1.5 });
        break;
      default: break;
    }
  }

  update(dt, t, center, env) {
    const U = this.U;
    // weather amounts
    const raining = env.weather === 1, snowing = raining && env.temp < 0;
    this.rainAmt = lerp(this.rainAmt, raining && !snowing ? 1 : 0, 1 - Math.exp(-dt * 0.8));
    this.snowAmt = lerp(this.snowAmt, snowing ? 1 : 0, 1 - Math.exp(-dt * 0.8));
    const s = U.uSeason.value;
    this.leafAmt = lerp(this.leafAmt, s.z * 0.9 + s.x * 0.6, 1 - Math.exp(-dt));
    const flyTarget = U.uNight.value * (s.x * 0.7 + s.y) * (1 - this.rainAmt) * (1 - this.snowAmt);
    this.flyAmt = lerp(this.flyAmt, flyTarget, 1 - Math.exp(-dt));
    for (const [o, amt] of [[this.rain, this.rainAmt], [this.snow, this.snowAmt], [this.leaves, this.leafAmt * 0.6]]) {
      const u = o.material.uniforms;
      u.uCenter.value.copy(center); u.uTime.value = t; u.uAmount.value = amt; o.visible = amt > 0.01;
    }
    const lu = this.leaves.material.uniforms;
    if (s.z >= s.x) { lu.uColA.value.setRGB(0.85, 0.3, 0.05); lu.uColB.value.setRGB(0.95, 0.65, 0.1); }
    else { lu.uColA.value.setRGB(1.0, 0.72, 0.82); lu.uColB.value.setRGB(1.0, 0.95, 0.97); }
    this.flies.material.uniforms.uAmount.value = this.flyAmt;
    this.flies.visible = this.flyAmt > 0.01;
    if (this.flies.visible && center.distanceTo(this.flyCenter) > 35) this._placeFlies(center);
    // fire: flames + smoke + up to 3 lights
    const W = this.world, cells = W.fireCells || [];
    let lit = 0;
    if (cells.length) {
      const budget = Math.min(110, 8 + cells.length * 0.4);
      const near = [];
      for (let k = 0; k < budget; k++) {
        const i = cells[(Math.random() * cells.length) | 0];
        const x = i % W.W, y = (i / W.W) | 0;
        const X = W.wx(x) + (Math.random() - 0.5), Z = W.wz(y) + (Math.random() - 0.5), Y = W.heightAt(x, y);
        if (Math.abs(X - center.x) > 120 || Math.abs(Z - center.z) > 120) continue;
        const R = Math.random();
        this.glow.emit(X, Y + 0.2, Z, (Math.random() - 0.5) * 0.4, 1.4 + R * 1.8, (Math.random() - 0.5) * 0.4, 0.45 + R * 0.5, 0.28 + R * 0.3,
          1.0, 0.22 + R * 0.35, 0.03, { drag: 0.6, grow: -0.6, alpha: 0.75 });
        if (Math.random() < 0.35) this.soft.emit(X, Y + 1.2, Z, 0.4, 1.0 + R, 0.15, 3 + R * 2, 0.7, 0.08, 0.075, 0.07, { drag: 0.2, grow: 2.5, alpha: 0.55 });
        if (Math.random() < 0.15) this.glow.emit(X, Y + 0.5, Z, (Math.random() - 0.5) * 2, 2 + R * 3, (Math.random() - 0.5) * 2, 1.5, 0.08, 1.0, 0.6, 0.2, { drag: 0.3 });
        near.push([X, Y, Z]);
      }
      for (let l = 0; l < 3 && near.length; l++) {
        const [X, Y, Z] = near[(l * near.length / 3) | 0];
        const L = this.fireLights[l];
        L.position.set(X, Y + 1.5, Z);
        L.intensity = (18 + 10 * Math.sin(t * 17 + l * 3)) * (0.4 + U.uNight.value);
        lit++;
      }
    }
    for (let l = lit; l < 3; l++) this.fireLights[l].intensity = 0;
    this.glow.update(dt);
    this.soft.update(dt);
    for (const r of this.rings) {
      if (!r.visible) continue;
      const u = r.userData;
      u.life -= dt;
      if (u.life <= 0) { r.visible = false; continue; }
      const k = 1 - u.life / u.max;
      r.scale.setScalar(0.3 + k * u.maxR);
      r.material.opacity = (1 - k) * 0.9;
    }
  }

  _placeFlies(center) {
    const W = this.world;
    this.flyCenter.copy(center);
    const n = this.flyBase.length / 3;
    let k = 0, tries = 0;
    while (k < n && tries < n * 20) {
      tries++;
      const X = center.x + (Math.random() - 0.5) * 120, Z = center.z + (Math.random() - 0.5) * 120;
      const [x, y] = W.cellOf(X, Z);
      if (x < 0 || y < 0 || x >= W.W || y >= W.H) continue;
      const t = W.terrain[y * W.W + x];
      if (t !== T.FOREST && t !== T.MEADOW && t !== T.GRASS) continue;
      this.flyBase.set([X, W.heightAt(x, y), Z], k * 3); k++;
    }
    this.flies.geometry.setDrawRange(0, k);
    this.flies.geometry.attributes.position.needsUpdate = true;
  }

  dispose() { this.scene.remove(this.group); this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); }); }
}
