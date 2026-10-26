// Sky dome, stars, moon, sun/moon light with shadows, hemisphere light; drives shared lighting uniforms.
import * as THREE from 'three';
import { smooth, lerp, clamp } from './util.js';

const C = (h) => new THREE.Color(h);
const MOON = C(0xa8bcff), NIGHT_SKY = C(0x3a4f8a), NIGHT_GROUND = C(0x141a2a);
// palettes keyed by sun elevation (sin of sun angle)
const KEYS = [
  { s: -0.35, zen: C(0x03060f), hor: C(0x0b1428), sun: C(0x6f8cff), amb: C(0x1a2440) },
  { s: -0.08, zen: C(0x101a3a), hor: C(0x6a3e58), sun: C(0xff7a4a), amb: C(0x2a2440) },
  { s: 0.05, zen: C(0x2b4a86), hor: C(0xf09a5a), sun: C(0xffa865), amb: C(0x4a4a5a) },
  { s: 0.25, zen: C(0x3d78cc), hor: C(0xb9d6ee), sun: C(0xfff0d8), amb: C(0x5f6f86) },
  { s: 1.0, zen: C(0x2f6fd0), hor: C(0xc4e0f6), sun: C(0xfff8ec), amb: C(0x6a7c94) },
];
function palette(s, out) {
  let i = 0;
  while (i < KEYS.length - 2 && s > KEYS[i + 1].s) i++;
  const a = KEYS[i], b = KEYS[i + 1];
  const t = smooth(a.s, b.s, s);
  for (const k of ['zen', 'hor', 'sun', 'amb']) out[k].copy(a[k]).lerp(b[k], t);
  return out;
}

export class Sky {
  constructor(scene, U, size) {
    this.U = U;
    this.scene = scene;
    this.size = size;
    this.pal = { zen: new THREE.Color(), hor: new THREE.Color(), sun: new THREE.Color(), amb: new THREE.Color() };
    // dome
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(1800, 32, 16), new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { uZenith: U.uZenith, uHorizon: U.uHorizon, uSunDir: { value: new THREE.Vector3() }, uSunColor: { value: new THREE.Color() },
        uMoonDir: { value: new THREE.Vector3() }, uNight: U.uNight, uCloud: U.uCloud, uTime: U.uTime, uFog: { value: 0 } },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); vec4 p = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*p; gl_Position.z = gl_Position.w; }`,
      fragmentShader: /* glsl */`
        uniform vec3 uZenith, uHorizon, uSunDir, uSunColor, uMoonDir; uniform float uNight, uCloud, uTime, uFog;
        varying vec3 vDir;
        float h(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float n2(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f); return mix(mix(h(i),h(i+vec2(1,0)),u.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),u.x),u.y); }
        void main(){
          vec3 d = normalize(vDir);
          float y = max(d.y, 0.0);
          vec3 col = mix(uHorizon, uZenith, pow(smoothstep(-0.05, 0.75, d.y), 0.7));
          col = mix(col, uHorizon * 0.55, smoothstep(0.0, -0.25, d.y));
          float sd = max(dot(d, uSunDir), 0.0);
          col += uSunColor * (pow(sd, 900.0) * 30.0 + pow(sd, 12.0) * 0.35 + pow(sd, 3.0) * 0.12) * (1.0 - uCloud * 0.75);
          float md = max(dot(d, uMoonDir), 0.0);
          col += vec3(0.75, 0.82, 1.0) * (smoothstep(0.9993, 0.9996, md) * 2.2 + pow(md, 60.0) * 0.12) * uNight * (1.0 - uCloud * 0.8);
          // clouds
          vec2 cp = d.xz / max(d.y, 0.08) * 1.6 + uTime * vec2(0.006, 0.002);
          float c = n2(cp) * 0.5 + n2(cp * 2.1) * 0.3 + n2(cp * 4.3) * 0.2;
          float cov = smoothstep(0.62 - uCloud * 0.45, 0.95, c) * smoothstep(0.0, 0.25, d.y);
          vec3 ccol = mix(uHorizon * 1.05 + vec3(0.06), vec3(0.5, 0.52, 0.56) * (1.0 - uNight * 0.85), uCloud * 0.7);
          col = mix(col, ccol, cov * 0.85);
          col = mix(col, uHorizon, uFog * 0.85);
          gl_FragColor = vec4(col, 1.0);
        }`,
    }));
    this.dome.renderOrder = -10;
    this.dome.frustumCulled = false;
    scene.add(this.dome);
    // stars
    const n = 2600, pos = new Float32Array(n * 3), sz = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, r = Math.sqrt(1 - u * u);
      pos.set([Math.cos(a) * r * 1700, Math.abs(u) * 1700, Math.sin(a) * r * 1700], i * 3);
      sz[i] = Math.random() ** 3 * 2.4 + 0.6;
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    sg.setAttribute('aSize', new THREE.BufferAttribute(sz, 1));
    this.stars = new THREE.Points(sg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uNight: U.uNight, uTime: U.uTime, uCloud: U.uCloud },
      vertexShader: `attribute float aSize; uniform float uTime; varying float vT;
        void main(){ vec4 p = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*p; gl_Position.z = gl_Position.w*0.9999;
          vT = 0.65 + 0.35*sin(uTime*2.0 + position.x*0.37 + position.z*0.11); gl_PointSize = aSize*1.6; }`,
      fragmentShader: `uniform float uNight, uCloud; varying float vT;
        void main(){ float d = length(gl_PointCoord-0.5); float a = smoothstep(0.5, 0.0, d); gl_FragColor = vec4(vec3(0.85,0.9,1.0)*a*vT*uNight*(1.-uCloud*0.9), 1.0); }`,
    }));
    this.stars.frustumCulled = false;
    this.stars.renderOrder = -9;
    scene.add(this.stars);
    // lights
    this.light = new THREE.DirectionalLight(0xffffff, 2.5);
    this.light.castShadow = true;
    this.light.shadow.mapSize.set(4096, 4096);
    this.light.shadow.bias = -0.0004;
    this.light.shadow.normalBias = 0.03;
    const sc = this.light.shadow.camera;
    sc.near = 1; sc.far = 900;
    scene.add(this.light, this.light.target);
    this.hemi = new THREE.HemisphereLight(0xbfd8ff, 0x3a3020, 0.7);
    scene.add(this.hemi);
    this.fog = new THREE.FogExp2(0x9fb8d0, 0.002);
    scene.fog = this.fog;
    this.sunDir = new THREE.Vector3();
    this.moonDir = new THREE.Vector3();
    this.cloud = 0; this.fogAmt = 0;
  }

  // tod in [0,1) (0.25 = sunrise at tick 0), weather 0 clear 1 rain 2 fog
  update(tod, weather, target, camDist, dt) {
    const U = this.U;
    const a = 2 * Math.PI * (tod - 0.25);
    const s = Math.sin(a);
    this.sunDir.set(Math.cos(a) * 0.82, s, 0.38).normalize();
    this.moonDir.set(-Math.cos(a) * 0.8, -s * 0.9 + 0.12, -0.42).normalize();
    const pal = palette(s, this.pal);
    const tCloud = weather === 1 ? 0.9 : weather === 2 ? 0.55 : 0.12;
    const tFog = weather === 2 ? 1 : 0;
    const k = 1 - Math.exp(-dt * 0.6);
    this.cloud = lerp(this.cloud, tCloud, k);
    this.fogAmt = lerp(this.fogAmt, tFog, k);
    const night = 1 - smooth(-0.22, 0.08, s);
    U.uNight.value = night;
    U.uCloud.value = this.cloud;
    // overcast desaturates the sky
    const grey = new THREE.Color().setRGB(0.42, 0.45, 0.5).multiplyScalar(1 - night * 0.85);
    U.uZenith.value.copy(pal.zen).lerp(grey, this.cloud * 0.65);
    U.uHorizon.value.copy(pal.hor).lerp(grey.clone().multiplyScalar(1.25), this.cloud * 0.6);
    this.dome.material.uniforms.uSunDir.value.copy(this.sunDir);
    this.dome.material.uniforms.uSunColor.value.copy(pal.sun);
    this.dome.material.uniforms.uMoonDir.value.copy(this.moonDir);
    this.dome.material.uniforms.uFog.value = this.fogAmt;
    // one shadow-casting light: sun by day, moon by night
    const day = smooth(-0.12, 0.12, s);
    const dir = day > 0.5 ? this.sunDir : this.moonDir;
    const dimming = 1 - this.cloud * 0.45;
    const sunI = smooth(-0.05, 0.3, s) * 2.8 * dimming;
    const moonI = 1.1 * night * (1 - this.cloud * 0.5);
    this.light.color.copy(day > 0.5 ? pal.sun : MOON);
    this.light.intensity = day > 0.5 ? sunI : moonI;
    this.light.position.copy(target).addScaledVector(dir, 400);
    this.light.target.position.copy(target);
    const half = clamp(camDist * 1.35, 45, Math.max(this.size, 60) * 0.75);
    const sc = this.light.shadow.camera;
    if (Math.abs(sc.right - half) > 2) { sc.left = -half; sc.right = half; sc.top = half; sc.bottom = -half; sc.updateProjectionMatrix(); }
    this.hemi.color.copy(pal.zen).lerp(pal.hor, 0.5).lerp(NIGHT_SKY, night);
    this.hemi.groundColor.setRGB(0.22, 0.18, 0.12).lerp(NIGHT_GROUND, night);
    this.hemi.intensity = lerp(1.05, 0.95, 1 - night) * (1 + this.cloud * 0.25);
    U.uSunDir.value.copy(dir);
    U.uSunColor.value.copy(this.light.color).multiplyScalar(this.light.intensity / 2.8);
    U.uAmbient.value.copy(this.hemi.color).multiplyScalar(this.hemi.intensity * 0.45);
    // fog
    const fogDensity = lerp(0.0016, 0.013, this.fogAmt) + this.cloud * 0.0012;
    this.fog.density = fogDensity;
    this.fog.color.copy(U.uHorizon.value).multiplyScalar(lerp(1.0, 0.8, night));
    U.uFogColor.value.copy(this.fog.color);
    U.uFogDensity.value = fogDensity;
    return { night, day, sunElev: s };
  }

  follow(camera) { this.dome.position.copy(camera.position); this.stars.position.copy(camera.position); }
}
