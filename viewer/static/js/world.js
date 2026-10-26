// Static world + slowly changing fields: terrain, water, ocean, trees, fruit, grass, flowers, rocks,
// carcasses/dropped food, built shelters. Visual relief is derived server-side (scene_heights); the
// simulation itself is a flat grid.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { T, TERRAIN_HEX, GLSL_NOISE, mulberry32, vnoise, hash2, clamp } from './util.js';

export function makeUniforms() {
  return {
    uTime: { value: 0 }, uField: { value: null }, uFlags: { value: null }, uOverlayTex: { value: null },
    uSeason: { value: new THREE.Vector4(1, 0, 0, 0) }, uDrought: { value: 0 }, uWet: { value: 0 }, uCloud: { value: 0 },
    uWind: { value: 0.6 }, uOverlay: { value: 0 }, uNight: { value: 0 }, uDetail: { value: 1 },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunColor: { value: new THREE.Color(1, 1, 1) },
    uZenith: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uAmbient: { value: new THREE.Color() },
    uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.002 },
  };
}

const COMMON_DECL = /* glsl */`
uniform float uTime; uniform sampler2D uField; uniform sampler2D uFlags; uniform vec4 uSeason; uniform float uWind;
${GLSL_NOISE}`;

function inject(mat, U, vertDecl, vertBody, fragDecl, fragColor, extra = {}) {
  mat.onBeforeCompile = (sh) => {
    for (const k of Object.keys(U)) sh.uniforms[k] = U[k];
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\n${COMMON_DECL}\n${vertDecl}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${vertBody}`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\n${COMMON_DECL}\n${fragDecl}`)
      .replace('#include <color_fragment>', `#include <color_fragment>\n${fragColor}`);
    if (extra.rough) sh.fragmentShader = sh.fragmentShader.replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>\n${extra.rough}`);
    if (extra.normal) sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${extra.normal}`);
    if (extra.emissive) sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${extra.emissive}`);
  };
  return mat;
}

export class World {
  constructor(scene, init, U) {
    this.scene = scene;
    this.U = U;
    this.W = init.W; this.H = init.H;
    const d = init.data;
    this.terrain = d.terrain; this.height = d.height; this.water = d.water; this.floodLvl = d.flood; this.fert = d.fertility;
    this.treeX = d.tree_x; this.treeY = d.tree_y; this.treeSeason = d.tree_season;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.rand = mulberry32(init.seed * 7919 + 17);
    this.fieldData = new Uint8Array(this.W * this.H * 4);
    this.texLin = this._dataTex(this.fieldData, THREE.LinearFilter);
    this.texNear = this._dataTex(this.fieldData, THREE.NearestFilter);
    this.densData = new Uint8Array(this.W * this.H * 2);      // R recent visits (overlay), G long-term trails
    this.densF = new Float32Array(this.W * this.H);
    this.trailF = new Float32Array(this.W * this.H);
    this.densTex = new THREE.DataTexture(this.densData, this.W, this.H, THREE.RGFormat, THREE.UnsignedByteType);
    this.densTex.magFilter = this.densTex.minFilter = THREE.LinearFilter;
    this.densTex.needsUpdate = true;
    U.uField.value = this.texLin; U.uFlags.value = this.texNear; U.uOverlayTex.value = this.densTex;
    this._buildTerrain();
    this._buildWater();
    this._buildTrees();
    this._buildGrass();
    this._buildRocks();
    this._buildFood();
  }

  _dataTex(data, filter) {
    const t = new THREE.DataTexture(data, this.W, this.H, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.magFilter = t.minFilter = filter;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.needsUpdate = true;
    return t;
  }

  // cell coordinates -> world
  wx(x) { return x - (this.W - 1) / 2; }
  wz(y) { return y - (this.H - 1) / 2; }
  cellOf(X, Z) { return [Math.round(X + (this.W - 1) / 2), Math.round(Z + (this.H - 1) / 2)]; }
  heightAt(x, y) {   // bilinear, x/y in (fractional) cell coordinates
    const W = this.W, H = this.H, h = this.height;
    x = clamp(x, 0, W - 1.001); y = clamp(y, 0, H - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const i = y0 * W + x0;
    const a = h[i], b = h[i + 1], c = h[i + W], e = h[i + W + 1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + e * fx) * fy;
  }
  surfaceAt(x, y) {   // ground or water surface (agents wade in shallow water)
    const g = this.heightAt(x, y);
    const xi = clamp(Math.round(x), 0, this.W - 1), yi = clamp(Math.round(y), 0, this.H - 1);
    const t = this.terrain[yi * this.W + xi];
    if (t === T.SHALLOW) { const wl = this.water[yi * this.W + xi]; if (wl === wl) return Math.max(g, wl - 0.28); }
    return g;
  }

  // ------------------------------------------------------------------ terrain
  _buildTerrain() {
    const { W, H, terrain, height, fert } = this;
    const n = W * H;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), info = new Float32Array(n * 4), cell = new Float32Array(n * 2);
    const pal = TERRAIN_HEX.map((h) => new THREE.Color(h));
    const VEG = [0, 0, 0.15, 1, 1, 0.8, 0.4, 0.2, 0, 0], CRACK = [0, 0, 0.3, 0.6, 0.4, 0.1, 1, 0.2, 0, 0];
    const MTN = [0, 0, 0, 0, 0, 0, 0, 0.35, 1, 0.5];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, t = terrain[i];
      pos[i * 3] = this.wx(x); pos[i * 3 + 1] = height[i]; pos[i * 3 + 2] = this.wz(y);
      const c = pal[t];
      const v = 0.86 + 0.22 * vnoise(x * 0.11, y * 0.11) + 0.06 * (hash2(x, y) - 0.5);
      const w = 0.04 * (vnoise(x * 0.05 + 40, y * 0.05) - 0.5);
      col[i * 3] = c.r * v * (1 + w); col[i * 3 + 1] = c.g * v; col[i * 3 + 2] = c.b * v * (1 - w);
      info[i * 4] = fert[i]; info[i * 4 + 1] = VEG[t]; info[i * 4 + 2] = CRACK[t]; info[i * 4 + 3] = MTN[t];
      cell[i * 2] = (x + 0.5) / W; cell[i * 2 + 1] = (y + 0.5) / H;
    }
    const idx = new Uint32Array((W - 1) * (H - 1) * 6);
    let k = 0;
    for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
      const a = y * W + x, b = a + 1, c = a + W, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = b; idx[k++] = b; idx[k++] = c; idx[k++] = d;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aInfo', new THREE.BufferAttribute(info, 4));
    g.setAttribute('aCell', new THREE.BufferAttribute(cell, 2));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    this.terrainGeo = g;
    const mat = inject(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.93, metalness: 0.0 }), this.U,
      `attribute vec4 aInfo; attribute vec2 aCell; varying vec4 vInfo; varying vec2 vCell; varying vec3 vWPos; varying vec3 vWNormal;`,
      `vInfo = aInfo; vCell = aCell; vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz; vWNormal = normalize(mat3(modelMatrix) * objectNormal);`,
      `uniform sampler2D uOverlayTex; uniform int uOverlay; uniform float uDrought; uniform float uWet; uniform float uCloud; uniform float uDetail;
       varying vec4 vInfo; varying vec2 vCell; varying vec3 vWPos; varying vec3 vWNormal;
       vec3 gRamp(float t){ t = clamp(t, 0., 1.); return t < 0.5 ? mix(vec3(0.02,0.03,0.18), vec3(0.05,0.55,0.35), t*2.) : mix(vec3(0.05,0.55,0.35), vec3(1.0,0.85,0.15), t*2.-1.); }`,
      /* glsl */`
      vec4 gFld = texture2D(uField, vCell);
      float gFa = texture2D(uFlags, vCell).a;
      float gFlood = gFlag(gFa, 1.), gBurnt = gFlag(gFa, 8.);
      float gFert = vInfo.x, gVeg = vInfo.y, gCrack = vInfo.z, gMtn = vInfo.w;
      vec3 gc = diffuseColor.rgb;
      float gPlantN = clamp(gFld.r / max(gFert, 0.04), 0., 1.);
      vec3 gStraw = vec3(0.30, 0.25, 0.12);
      gc = mix(gc, mix(gStraw, gc * 0.8, 0.3), gVeg * (1. - smoothstep(0.08, 0.75, gPlantN)) * 0.85);
      float gLum = dot(gc, vec3(0.3, 0.59, 0.11));
      gc = mix(gc, gc * vec3(0.96, 1.08, 0.88), uSeason.x * gVeg * 0.5);
      gc = mix(gc, gc * vec3(1.15, 0.95, 0.68), uSeason.z * gVeg * 0.45);
      gc = mix(gc, vec3(gLum) * vec3(0.98, 0.9, 0.82), uSeason.w * gVeg * 0.55);
      gc = mix(gc, gc * vec3(1.1, 0.98, 0.78), uDrought * gVeg * 0.5);
      float gCr = max(1. - smoothstep(0.0, 0.045, abs(gNoise(vWPos.xz * 0.85) - 0.5)),
                      1. - smoothstep(0.0, 0.045, abs(gNoise(vWPos.zx * 1.3 + 9.) - 0.5)));
      gc *= 1. - 0.55 * gCr * uDrought * gCrack;
      float gSlope = 1. - clamp(vWNormal.y, 0., 1.);
      gc = mix(gc, vec3(0.16, 0.15, 0.135), smoothstep(0.32, 0.62, gSlope) * 0.8);
      float gCap = gMtn * smoothstep(27.5, 30.5, vWPos.y + gNoise(vWPos.xz * 0.35) * 3.);
      float gSnow = max(smoothstep(0.05, 0.45, gFld.g), gCap) * (1. - smoothstep(0.6, 0.95, gSlope) * 0.6);
      gc = mix(gc, vec3(0.86, 0.9, 0.97), gSnow);
      gc = mix(gc, vec3(0.03, 0.026, 0.024), gBurnt * 0.85);
      gc = mix(gc, gc * vec3(0.5, 0.45, 0.36), gFlood * 0.7);
      // footpaths: ground worn where agents have walked often (long-term visit counts; visual only)
      float gTrail = smoothstep(0.08, 0.7, texture2D(uOverlayTex, vCell).g) * gVeg * (1. - gSnow);
      gc = mix(gc, vec3(0.2, 0.15, 0.09) * (0.8 + 0.4 * gNoise(vWPos.xz * 3.1)), gTrail * 0.55);
      gc *= 1. - 0.16 * uWet * (1. - gSnow);
      float gCs = smoothstep(0.5, 0.78, gFbm(vWPos.xz * 0.011 + uTime * vec2(0.012, 0.005))) * uCloud;
      gc *= 1. - 0.28 * gCs;
      // overlays are partly self-lit so they stay readable at night
      vec3 gOv = vec3(0.0); float gOvA = 0.0;
      if (uOverlay == 1) { gOv = gRamp(gFld.r * 1.1); gOvA = 0.75; }
      else if (uOverlay == 2) { float dn = texture2D(uOverlayTex, vCell).r; gOv = gRamp(dn * 1.4); gOvA = smoothstep(0.0, 0.05, dn) * 0.85; gc *= 0.5; }
      else if (uOverlay == 3) { gOv = gRamp(gFert); gOvA = 0.75; }
      gc = mix(gc, gOv, gOvA);
      diffuseColor.rgb = gc;
      float gFireAmt = gFld.b;
      `,
      {
        rough: `roughnessFactor = mix(roughnessFactor, 0.32, clamp(uWet * 0.85 + gFlood, 0., 1.) * (1. - gSnow));`,
        normal: `if (uDetail > 0.5) { vec2 nq = vWPos.xz * 2.3; float b0 = gFbm(nq), bx = gFbm(nq + vec2(0.07, 0.)), bz = gFbm(nq + vec2(0., 0.07));
                   float nb = gNoise(vWPos.xz * 9.0) - gNoise(vWPos.xz * 9.0 + vec2(0.05, 0.0));
                   vec3 bw = vec3((b0 - bx) * 0.9 + nb * 0.25, 0.0, (b0 - bz) * 0.9) * (1.0 - gSnow * 0.7);
                   normal = normalize(normal + (viewMatrix * vec4(bw, 0.0)).xyz); }`,
        emissive: `totalEmissiveRadiance += gOv * gOvA * 0.35;
          totalEmissiveRadiance += vec3(1.0, 0.22, 0.02) * gFireAmt * (0.9 + 0.5 * sin(uTime * 13. + vWPos.x * 2.1 + vWPos.z * 1.3));`,
      });
    this.terrainMesh = new THREE.Mesh(g, mat);
    this.terrainMesh.receiveShadow = true;
    this.terrainMesh.castShadow = true;
    this.group.add(this.terrainMesh);
  }

  // ------------------------------------------------------------------ water
  _waterMaterial(ring) {
    const U = this.U;
    return new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      defines: ring ? { RING: 1 } : {},
      uniforms: {
        uTime: U.uTime, uFlags: U.uFlags, uSunDir: U.uSunDir, uSunColor: U.uSunColor, uZenith: U.uZenith,
        uHorizon: U.uHorizon, uAmbient: U.uAmbient, uFogColor: U.uFogColor, uFogDensity: U.uFogDensity, uWet: U.uWet,
        uMask: { value: new THREE.Vector2(this.W / 2 - 0.5, this.H / 2 - 0.5) },
      },
      vertexShader: /* glsl */`
        attribute float aLevel; attribute float aFlood; attribute float aGround; attribute vec2 aCell;
        uniform sampler2D uFlags; uniform float uTime;
        varying vec3 vWPos; varying float vDepth;
        ${GLSL_NOISE}
        void main() {
          float fa = texture2D(uFlags, aCell).a;
          float y = aLevel > -900. ? aLevel : aGround - 3.0;
          if (aLevel > 0.01 && gFlag(fa, 2.) > 0.5) y = aGround - 1.2;         // stream dried out (drought)
          if (aFlood > -900. && gFlag(fa, 1.) > 0.5) y = max(y, aFlood);      // floodplain under water
          vec3 p = vec3(position.x, y, position.z);
          p.y += 0.035 * sin(uTime * 1.3 + p.x * 0.45 + p.z * 0.31);
          vDepth = y - aGround;
          vec4 wp = modelMatrix * vec4(p, 1.0);
          vWPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: /* glsl */`
        uniform float uTime; uniform vec3 uSunDir; uniform vec3 uSunColor; uniform vec3 uZenith; uniform vec3 uHorizon;
        uniform vec3 uAmbient; uniform vec3 uFogColor; uniform float uFogDensity; uniform float uWet; uniform vec2 uMask;
        varying vec3 vWPos; varying float vDepth;
        ${GLSL_NOISE}
        float wh(vec2 p, float t) { return gNoise(p * 0.8 + vec2(t * 0.32, t * 0.18)) + 0.5 * gNoise(p * 2.1 - vec2(t * 0.45, -t * 0.28))
                                         + uWet * 0.35 * gNoise(p * 7.0 + t * 2.0); }
        void main() {
          #ifdef RING
          if (abs(vWPos.x) < uMask.x && abs(vWPos.z) < uMask.y) discard;
          #endif
          if (vDepth < 0.015) discard;
          vec2 p = vWPos.xz; float t = uTime; float e = 0.12;
          float h0 = wh(p, t), hx = wh(p + vec2(e, 0.), t), hz = wh(p + vec2(0., e), t);
          vec3 n = normalize(vec3((h0 - hx) * 1.4, 1.0, (h0 - hz) * 1.4));
          vec3 V = normalize(cameraPosition - vWPos);
          float fres = 0.03 + 0.97 * pow(1. - max(dot(n, V), 0.), 5.);
          vec3 R = reflect(-V, n);
          vec3 sky = mix(uHorizon, uZenith, smoothstep(0.0, 0.6, R.y));
          float d = smoothstep(0.0, 3.2, vDepth);
          vec3 lightc = max(uAmbient + uSunColor * max(uSunDir.y, 0.0) * 0.55, vec3(0.16, 0.2, 0.26));
          vec3 body = mix(vec3(0.09, 0.34, 0.32), vec3(0.012, 0.06, 0.11), d) * lightc;
          vec3 col = mix(body, sky, fres * 0.85);
          float sd = max(dot(R, uSunDir), 0.0);
          col += uSunColor * (pow(sd, 220.) * 5.0 + pow(sd, 22.) * 0.12) * (1. - uWet * 0.7);
          float foam = (1. - smoothstep(0.0, 0.12, vDepth)) * smoothstep(0.45, 0.85, gNoise(p * 3.2 + t * 0.7));
          col = mix(col, vec3(0.9, 0.95, 1.0) * lightc * 1.2, foam * 0.55);
          float alpha = clamp(0.55 + d * 0.42 + fres * 0.35 + foam, 0., 0.97);
          float dist = length(vWPos - cameraPosition);
          col = mix(col, uFogColor, 1. - exp(-uFogDensity * uFogDensity * dist * dist));
          gl_FragColor = vec4(col, alpha);
        }`,
    });
  }

  _buildWater() {
    const { W, H, water, floodLvl, height } = this;
    const n = W * H;
    const keep = new Uint8Array(n);
    for (let i = 0; i < n; i++) keep[i] = (water[i] === water[i]) || (floodLvl[i] === floodLvl[i]) ? 1 : 0;
    const pos = this.terrainGeo.attributes.position.array;
    const lvl = new Float32Array(n), fl = new Float32Array(n), gr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      lvl[i] = water[i] === water[i] ? water[i] : -999;
      fl[i] = floodLvl[i] === floodLvl[i] ? floodLvl[i] : -999;
      gr[i] = height[i];
    }
    const idx = [];
    for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
      const a = y * W + x, b = a + 1, c = a + W, d = c + 1;
      if (keep[a] || keep[b] || keep[c] || keep[d]) idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aLevel', new THREE.BufferAttribute(lvl, 1));
    g.setAttribute('aFlood', new THREE.BufferAttribute(fl, 1));
    g.setAttribute('aGround', new THREE.BufferAttribute(gr, 1));
    g.setAttribute('aCell', this.terrainGeo.attributes.aCell);
    g.setIndex(idx);
    this.waterMesh = new THREE.Mesh(g, this._waterMaterial(false));
    this.waterMesh.renderOrder = 2;
    this.waterMesh.frustumCulled = false;
    this.group.add(this.waterMesh);
    // ocean beyond the map edge
    const S = 3000, N = 24;
    const rg = new THREE.PlaneGeometry(S, S, N, N).rotateX(-Math.PI / 2);
    const m = rg.attributes.position.count;
    rg.setAttribute('aLevel', new THREE.BufferAttribute(new Float32Array(m).fill(0), 1));
    rg.setAttribute('aFlood', new THREE.BufferAttribute(new Float32Array(m).fill(-999), 1));
    rg.setAttribute('aGround', new THREE.BufferAttribute(new Float32Array(m).fill(-14), 1));
    rg.setAttribute('aCell', new THREE.BufferAttribute(new Float32Array(m * 2).fill(0.001), 2));
    this.ring = new THREE.Mesh(rg, this._waterMaterial(true));
    this.ring.renderOrder = 1;
    this.group.add(this.ring);
  }

  // ------------------------------------------------------------------ flora
  _part(geo, foliage) {
    geo = geo.index ? geo.toNonIndexed() : geo;
    geo.deleteAttribute('uv');
    geo.setAttribute('aFoliage', new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count).fill(foliage), 1));
    return geo;
  }

  _buildTrees() {
    const { W, H, terrain, height } = this;
    const R = this.rand;
    const conifer = mergeGeometries([
      this._part(new THREE.CylinderGeometry(0.05, 0.09, 0.7, 6).translate(0, 0.35, 0), 0),
      ...[[0.6, 0.62, 0.7], [0.52, 0.58, 1.0], [0.43, 0.54, 1.3], [0.33, 0.5, 1.6], [0.22, 0.46, 1.88]].map(([r, h, y]) =>
        this._part(new THREE.ConeGeometry(r, h, 8, 1, true).translate(0, y, 0), 1)),
    ]);
    const blob = (r, x, y, z, seed, detail = 1) => {   // lumpy, smooth-shaded canopy cluster
      const g = new THREE.IcosahedronGeometry(r, detail);
      const pos = g.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const vx = pos.getX(i), vy = pos.getY(i), vz = pos.getZ(i);
        const n = 1 + 0.16 * Math.sin(vx * 9 + seed) * Math.sin(vy * 7 + seed * 2) * Math.sin(vz * 8 + seed * 3);
        pos.setXYZ(i, vx * n, vy * n * 0.88, vz * n);
      }
      g.computeVertexNormals();
      return this._part(g.translate(x, y, z), 1);
    };
    const broad = mergeGeometries([
      this._part(new THREE.CylinderGeometry(0.06, 0.12, 0.95, 7).translate(0, 0.47, 0), 0),
      this._part(new THREE.CylinderGeometry(0.03, 0.05, 0.5, 5).rotateZ(0.7).translate(0.2, 0.95, 0), 0),
      blob(0.6, 0, 1.2, 0, 1, 1), blob(0.42, 0.36, 1.0, 0.14, 2, 0), blob(0.4, -0.32, 1.05, -0.18, 3, 0),
    ]);
    const fruitTrees = new Map();
    for (let k = 0; k < this.treeX.length; k++) fruitTrees.set(this.treeY[k] * W + this.treeX[k], k);
    const items = { c: [], b: [] };
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, t = terrain[i];
      const fk = fruitTrees.get(i);
      if (fk !== undefined) { items.b.push([x, y, fk]); continue; }
      if (t === T.FOREST && R() < 0.62) (height[i] > 13 || R() < 0.3 ? items.c : items.b).push([x, y, -1]);
      else if ((t === T.MEADOW || t === T.GRASS) && R() < 0.012) items.b.push([x, y, -1]);
    }
    this.trees = [];
    this.fruitTreeInstance = new Int32Array(this.treeX.length).fill(-1);
    for (const [key, geo, kind] of [['c', conifer, 0], ['b', broad, 1]]) {
      const list = items[key];
      const n = list.length;
      const cellA = new Float32Array(n * 2), seedA = new Float32Array(n), kindA = new Float32Array(n).fill(kind), deadA = new Float32Array(n);
      const g = geo.clone();
      g.setAttribute('aCell', new THREE.InstancedBufferAttribute(cellA, 2));
      g.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seedA, 1));
      g.setAttribute('aKind', new THREE.InstancedBufferAttribute(kindA, 1));
      g.setAttribute('aDead', new THREE.InstancedBufferAttribute(deadA, 1));
      const mesh = new THREE.InstancedMesh(g, this._treeMaterial(), n);
      const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
      list.forEach(([x, y, fk], j) => {
        const jx = (R() - 0.5) * 0.5, jy = (R() - 0.5) * 0.5;
        const sc = (kind === 0 ? 1.05 + R() * 0.65 : 0.95 + R() * 0.5) * (fk >= 0 ? 1.1 : 1);
        p.set(this.wx(x + jx), this.heightAt(x + jx, y + jy) - 0.05, this.wz(y + jy));
        q.setFromAxisAngle(up, R() * Math.PI * 2);
        s.set(sc, sc * (0.9 + R() * 0.25), sc);
        mesh.setMatrixAt(j, m.compose(p, q, s));
        cellA[j * 2] = (x + 0.5) / W; cellA[j * 2 + 1] = (y + 0.5) / H;
        seedA[j] = R();
        if (fk >= 0) this.fruitTreeInstance[fk] = j;
      });
      mesh.castShadow = true; mesh.receiveShadow = true;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      this.group.add(mesh);
      this.trees.push(mesh);
      if (kind === 1) { this.broadMesh = mesh; this.broadDead = deadA; this.broadDeadAttr = g.attributes.aDead; }
    }
    // fruit: 5 per fruit tree on the canopy
    const nf = this.treeX.length * 5;
    const fg = new THREE.IcosahedronGeometry(0.1, 1);
    this.fruitMesh = new THREE.InstancedMesh(fg, new THREE.MeshStandardMaterial({ roughness: 0.4, emissive: 0x220500, emissiveIntensity: 0.6 }), Math.max(1, nf));
    this.fruitMesh.castShadow = false;
    this.fruitPos = new Float32Array(nf * 3);
    const FRUIT_COL = [new THREE.Color(0xff8fb1), new THREE.Color(0xd8221a), new THREE.Color(0xffa020)];
    const tm = new THREE.Matrix4(), tp = new THREE.Vector3(), tq = new THREE.Quaternion(), ts = new THREE.Vector3();
    for (let k = 0; k < this.treeX.length; k++) {
      const j = this.fruitTreeInstance[k];
      this.broadMesh.getMatrixAt(Math.max(0, j), tm);
      tm.decompose(tp, tq, ts);
      for (let f = 0; f < 5; f++) {
        const a = f / 5 * Math.PI * 2 + k, r = 0.5 * ts.x;
        const v = [tp.x + Math.cos(a) * r, tp.y + (0.95 + 0.35 * ((f * 7) % 3) / 2) * ts.y, tp.z + Math.sin(a) * r];
        this.fruitPos.set(v, (k * 5 + f) * 3);
        this.fruitMesh.setColorAt(k * 5 + f, FRUIT_COL[this.treeSeason[k]] || FRUIT_COL[1]);
        this.fruitMesh.setMatrixAt(k * 5 + f, new THREE.Matrix4().makeScale(0, 0, 0));
      }
    }
    if (this.fruitMesh.instanceColor) this.fruitMesh.instanceColor.needsUpdate = true;
    this.group.add(this.fruitMesh);
  }

  _treeMaterial() {
    return inject(new THREE.MeshStandardMaterial({ roughness: 0.85, side: THREE.DoubleSide }), this.U,
      `attribute float aFoliage; attribute vec2 aCell; attribute float aSeed; attribute float aKind; attribute float aDead;
       varying float vFol; varying float vSnow; varying float vBurnt; varying float vSeed; varying float vKind;`,
      /* glsl */`
       vec4 tFld = texture2D(uField, aCell); float tFa = texture2D(uFlags, aCell).a;
       float tBurnt = max(gFlag(tFa, 8.), aDead);
       vFol = aFoliage; vSeed = aSeed; vKind = aKind; vBurnt = tBurnt;
       vSnow = smoothstep(0.05, 0.4, tFld.g) * step(0.2, objectNormal.y) * aFoliage;
       float tShrink = mix(1.0, 0.3, tBurnt);
       if (aKind > 0.5) tShrink *= mix(1.0, 0.75, uSeason.w);
       vec3 tC = vec3(0.0, 0.55, 0.0);
       transformed = mix(transformed, tC + (transformed - tC) * tShrink, aFoliage);
       vec3 tIp = vec3(instanceMatrix[3]);
       float tSw = sin(uTime * 1.5 + aSeed * 6.283 + tIp.x * 0.17 + tIp.z * 0.11) * 0.6 + sin(uTime * 2.9 + aSeed * 3.1) * 0.25;
       float tH = transformed.y * transformed.y;
       transformed.x += tSw * uWind * 0.05 * tH;
       transformed.z += tSw * uWind * 0.025 * tH;`,
      `varying float vFol; varying float vSnow; varying float vBurnt; varying float vSeed; varying float vKind;`,
      /* glsl */`
       vec3 tTrunk = vec3(0.13, 0.075, 0.04);
       vec3 tLeaf;
       if (vKind < 0.5) tLeaf = mix(vec3(0.022, 0.085, 0.045), vec3(0.04, 0.14, 0.06), vSeed);
       else {
         vec3 sp = mix(vec3(0.16, 0.42, 0.07), vec3(0.42, 0.52, 0.1), vSeed * 0.5);
         vec3 su = mix(vec3(0.05, 0.21, 0.04), vec3(0.1, 0.3, 0.06), vSeed);
         vec3 au = mix(vec3(0.62, 0.17, 0.02), vec3(0.85, 0.5, 0.04), vSeed);
         vec3 wi = vec3(0.13, 0.1, 0.075);
         tLeaf = sp * uSeason.x + su * uSeason.y + au * uSeason.z + wi * uSeason.w;
       }
       vec3 tc = mix(tTrunk, tLeaf, vFol) * (0.85 + 0.3 * fract(vSeed * 7.31));
       tc = mix(tc, vec3(0.035, 0.03, 0.025), vBurnt);
       tc = mix(tc, vec3(0.85, 0.9, 0.96), vSnow * (1. - vBurnt));
       diffuseColor.rgb = tc;`);
  }

  _buildGrass() {
    const { W, H, terrain, fert } = this;
    const R = this.rand;
    const blades = [];
    for (let b = 0; b < 5; b++) {
      const a = b / 5 * Math.PI * 2 + 0.4 + (b % 2) * 0.3, r = 0.05 + 0.04 * (b % 3);
      const h = 0.34 + 0.06 * ((b * 5) % 4), lean = 0.12 + 0.05 * (b % 3);
      const cx = Math.cos(a) * r, cz = Math.sin(a) * r, dx = Math.cos(a), dz = Math.sin(a);
      const px = -Math.sin(a) * 0.016, pz = Math.cos(a) * 0.016;
      const mx = cx + dx * lean * 0.35, mz = cz + dz * lean * 0.35, my = h * 0.55;
      const tx = cx + dx * lean, tz = cz + dz * lean;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
        cx - px, 0, cz - pz, cx + px, 0, cz + pz, mx + px * 0.7, my, mz + pz * 0.7,
        cx - px, 0, cz - pz, mx + px * 0.7, my, mz + pz * 0.7, mx - px * 0.7, my, mz - pz * 0.7,
        mx - px * 0.7, my, mz - pz * 0.7, mx + px * 0.7, my, mz + pz * 0.7, tx, h, tz]), 3));
      g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(27).fill(0).map((_, i) => (i % 3 === 1 ? 1 : 0)), 3));
      blades.push(g);
    }
    const geo = mergeGeometries(blades);
    const list = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const t = terrain[y * W + x];
      const p = t === T.MEADOW ? 0.95 : t === T.GRASS ? 0.6 : t === T.FOREST ? 0.12 : t === T.BARREN ? 0.08 : 0;
      for (let k = 0; k < 2; k++) if (R() < p) list.push([x + (R() - 0.5) * 0.95, y + (R() - 0.5) * 0.95, t]);
    }
    const n = list.length;
    const cellA = new Float32Array(n * 2), fertA = new Float32Array(n), seedA = new Float32Array(n);
    geo.setAttribute('aCell', new THREE.InstancedBufferAttribute(cellA, 2));
    geo.setAttribute('aFert', new THREE.InstancedBufferAttribute(fertA, 1));
    geo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seedA, 1));
    const mat = inject(new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0.9 }), this.U,
      `attribute vec2 aCell; attribute float aFert; attribute float aSeed; varying float vPl; varying float vSeed; varying float vY;`,
      /* glsl */`
       vec4 gF = texture2D(uField, aCell);
       float gP = clamp(gF.r / max(aFert, 0.04), 0., 1.);
       vPl = gP; vSeed = aSeed; vY = transformed.y;
       float gh = (0.25 + 0.95 * gP) * (1. - 0.55 * uSeason.w) * (1. - smoothstep(0.2, 0.5, gF.g)) * (1. - gF.b);
       transformed.y *= gh;
       transformed.xz *= mix(0.6, 1.0, gh);
       vec3 gIp = vec3(instanceMatrix[3]);
       float gSw = sin(uTime * 2.2 + gIp.x * 0.35 + gIp.z * 0.21 + aSeed * 4.) * uWind;
       transformed.x += gSw * 0.12 * transformed.y * 2.0;
       transformed.z += gSw * 0.05 * transformed.y * 2.0;`,
      `varying float vPl; varying float vSeed; varying float vY; uniform float uDrought;`,
      /* glsl */`
       vec3 gLush = mix(vec3(0.09, 0.27, 0.04), vec3(0.2, 0.38, 0.07), vSeed);
       vec3 gDry = vec3(0.42, 0.32, 0.12);
       vec3 gcol = mix(gDry, gLush, smoothstep(0.1, 0.7, vPl) * (1. - uDrought * 0.6));
       gcol = mix(gcol, gcol * vec3(1.3, 0.85, 0.45), uSeason.z * 0.6);
       gcol = mix(gcol, vec3(0.32, 0.27, 0.2), uSeason.w * 0.6);
       gcol *= 0.45 + 1.0 * clamp(vY * 2.6, 0., 1.);
       gcol = mix(gcol, gcol * vec3(1.25, 1.2, 0.8), smoothstep(0.25, 0.45, vY) * 0.5);   // sunlit, drier tips
       diffuseColor.rgb = gcol;`);
    this.grass = new THREE.InstancedMesh(geo, mat, n);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    list.forEach(([x, y], j) => {
      p.set(this.wx(x), this.heightAt(x, y) - 0.02, this.wz(y));
      q.setFromAxisAngle(up, R() * 6.283);
      const sc = 0.8 + R() * 0.7;
      s.set(sc, sc * (0.8 + R() * 0.5), sc);
      this.grass.setMatrixAt(j, m.compose(p, q, s));
      const xi = clamp(Math.round(x), 0, W - 1), yi = clamp(Math.round(y), 0, H - 1);
      cellA[j * 2] = (xi + 0.5) / W; cellA[j * 2 + 1] = (yi + 0.5) / H;
      fertA[j] = fert[yi * W + xi]; seedA[j] = R();
    });
    this.grass.receiveShadow = true;
    this.grass.computeBoundingSphere();
    this.group.add(this.grass);

    // meadow flowers (spring/summer, need standing plants)
    const fl = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const t = terrain[y * W + x];
      if ((t === T.MEADOW && R() < 0.5) || (t === T.GRASS && R() < 0.06)) fl.push([x + (R() - 0.5) * 0.9, y + (R() - 0.5) * 0.9]);
    }
    const fgeo = new THREE.OctahedronGeometry(0.07, 0);
    const fcell = new Float32Array(fl.length * 2), ffert = new Float32Array(fl.length);
    fgeo.setAttribute('aCell', new THREE.InstancedBufferAttribute(fcell, 2));
    fgeo.setAttribute('aFert', new THREE.InstancedBufferAttribute(ffert, 1));
    const fmat = inject(new THREE.MeshStandardMaterial({ roughness: 0.6, emissive: 0x111111 }), this.U,
      `attribute vec2 aCell; attribute float aFert;`,
      /* glsl */`
       vec4 fF = texture2D(uField, aCell);
       float fP = clamp(fF.r / max(aFert, 0.04), 0., 1.);
       float fS = smoothstep(0.35, 0.7, fP) * (uSeason.x + uSeason.y * 0.7) * (1. - smoothstep(0.1, 0.3, fF.g)) * (1. - fF.b);
       transformed *= fS;`, '', '');
    this.flowers = new THREE.InstancedMesh(fgeo, fmat, Math.max(1, fl.length));
    const FC = [0xffe066, 0xffffff, 0xc58cff, 0xff8fc8, 0x8fd0ff].map((h) => new THREE.Color(h));
    fl.forEach(([x, y], j) => {
      p.set(this.wx(x), this.heightAt(x, y) + 0.32, this.wz(y));
      this.flowers.setMatrixAt(j, m.compose(p, q.identity(), s.set(1, 1, 1)));
      this.flowers.setColorAt(j, FC[Math.floor(R() * FC.length)]);
      const xi = clamp(Math.round(x), 0, W - 1), yi = clamp(Math.round(y), 0, H - 1);
      fcell[j * 2] = (xi + 0.5) / W; fcell[j * 2 + 1] = (yi + 0.5) / H; ffert[j] = fert[yi * W + xi];
    });
    this.flowers.computeBoundingSphere();
    this.group.add(this.flowers);
  }

  _buildRocks() {
    const { W, H, terrain } = this;
    const R = this.rand;
    const list = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const t = terrain[y * W + x];
      if (t === T.ROCK && R() < 0.09) list.push([x, y, 0.25 + R() * 0.35, 0]);
      else if (t === T.MOUNTAIN && R() < 0.035) list.push([x, y, 0.6 + R() * 0.8, 0]);
      else if (t === T.CAVE) list.push([x, y, 1.0, 1]);
      else if ((t === T.BARREN || t === T.SAND) && R() < 0.03) list.push([x, y, 0.15 + R() * 0.2, 0]);
    }
    const geo = new THREE.DodecahedronGeometry(1, 0);
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ flatShading: true, roughness: 0.95 }), Math.max(1, list.length));
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), c = new THREE.Color();
    const caves = [];
    list.forEach(([x, y, sc, cave], j) => {
      const jx = (R() - 0.5) * 0.6, jy = (R() - 0.5) * 0.6;
      p.set(this.wx(x + jx), this.heightAt(x + jx, y + jy) + sc * 0.25, this.wz(y + jy));
      q.setFromEuler(new THREE.Euler(R() * 3, R() * 3, R() * 3));
      s.set(sc * (0.8 + R() * 0.5), sc * (0.55 + R() * 0.4), sc * (0.8 + R() * 0.5));
      mesh.setMatrixAt(j, m.compose(p, q, s));
      const g = cave ? 0.12 : 0.33 + R() * 0.2;
      mesh.setColorAt(j, c.setRGB(g, g * 0.97, g * 0.92));
      if (cave) caves.push([x, y]);
    });
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.computeBoundingSphere();
    this.group.add(mesh);
    // cave mouths: dark arches with a faint warm interior (shelter)
    const cg = new THREE.TorusGeometry(0.45, 0.16, 6, 10, Math.PI);
    const cm = new THREE.InstancedMesh(cg, new THREE.MeshStandardMaterial({ color: 0x2a2420, flatShading: true, emissive: 0x1a0c04 }), Math.max(1, caves.length));
    caves.forEach(([x, y], j) => {
      p.set(this.wx(x), this.heightAt(x, y) + 0.05, this.wz(y));
      cm.setMatrixAt(j, m.compose(p, q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), R() * 6.28), s.set(1, 1, 1)));
    });
    cm.computeBoundingSphere();
    this.group.add(cm);
  }

  _buildFood() {
    const meat = mergeGeometries([
      new THREE.SphereGeometry(0.16, 8, 6).scale(1.3, 0.55, 1).translate(0, 0.08, 0),
      new THREE.CylinderGeometry(0.03, 0.03, 0.42, 5).rotateZ(Math.PI / 2).translate(0, 0.12, 0),
    ].map((g) => { g.deleteAttribute('uv'); return g.index ? g.toNonIndexed() : g; }));
    this.foodMesh = new THREE.InstancedMesh(meat, new THREE.MeshStandardMaterial({ color: 0x8c2a1c, roughness: 0.55, emissive: 0x200402 }), 3000);
    this.foodMesh.count = 0;
    this.foodMesh.castShadow = true;
    this.foodMesh.frustumCulled = false;
    this.group.add(this.foodMesh);
    const hut = new THREE.ConeGeometry(0.55, 0.8, 6).translate(0, 0.4, 0);
    this.builtMesh = new THREE.InstancedMesh(hut, new THREE.MeshStandardMaterial({ color: 0x7a5a3a, flatShading: true }), 2000);
    this.builtMesh.count = 0;
    this.builtMesh.castShadow = true;
    this.builtMesh.frustumCulled = false;
    this.group.add(this.builtMesh);
  }

  // ------------------------------------------------------------------ dynamic fields
  updateFields(msg) {
    const f = msg.data.field;
    this.fieldData.set(f);
    this.texLin.needsUpdate = true;
    this.texNear.needsUpdate = true;
    const { W, H } = this;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1), p = new THREE.Vector3();
    let nf = 0, nb = 0;
    this.fireCells = [];
    this.floodCount = 0;
    for (let i = 0, n = W * H; i < n; i++) {
      const a = f[i * 4 + 3];
      if (f[i * 4 + 2] > 0) this.fireCells.push(i);
      if (a === 0) continue;
      if (a & 1) this.floodCount++;
      if ((a & 4) && nf < 3000) {
        const x = i % W, y = (i / W) | 0;
        p.set(this.wx(x), this.heightAt(x, y), this.wz(y));
        q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), hash2(x, y) * 6.28);
        this.foodMesh.setMatrixAt(nf++, m.compose(p, q, s));
      }
      if ((a & 16) && nb < 2000) {
        const x = i % W, y = (i / W) | 0;
        p.set(this.wx(x), this.heightAt(x, y), this.wz(y));
        this.builtMesh.setMatrixAt(nb++, m.compose(p, q.identity(), s));
      }
    }
    this.foodMesh.count = nf; this.foodMesh.instanceMatrix.needsUpdate = true;
    this.builtMesh.count = nb; this.builtMesh.instanceMatrix.needsUpdate = true;
    // fruit
    const fruit = msg.data.fruit, alive = msg.data.tree_alive;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let k = 0; k < fruit.length; k++) {
      const show = alive && alive.length ? alive[k] : 1;
      const nShow = show ? Math.round(fruit[k] / 255 * 5) : 0;
      for (let j = 0; j < 5; j++) {
        const o = (k * 5 + j) * 3;
        if (j < nShow) { p.set(this.fruitPos[o], this.fruitPos[o + 1], this.fruitPos[o + 2]); this.fruitMesh.setMatrixAt(k * 5 + j, m.compose(p, q.identity(), s)); }
        else this.fruitMesh.setMatrixAt(k * 5 + j, zero);
      }
      const inst = this.fruitTreeInstance[k];
      if (inst >= 0 && this.broadDead) this.broadDead[inst] = show ? 0 : 1;
    }
    this.fruitMesh.instanceMatrix.needsUpdate = true;
    if (this.broadDeadAttr) this.broadDeadAttr.needsUpdate = true;
  }

  // agent-density overlay: exponentially decaying visit counts
  updateDensity(agents, n) {
    const d = this.densF, tr = this.trailF, W = this.W;
    for (let k = 0; k < n; k++) {
      const i = agents[k * 8 + 1] * W + agents[k * 8];
      d[i] += 1; tr[i] += 1;
    }
    this._densTicks = (this._densTicks || 0) + 1;
    const now = performance.now();
    if (now - (this._densT || 0) < 250) return;              // decay + texture upload at most 4x per second
    this._densT = now;
    const kd = Math.pow(0.985, this._densTicks), kt = Math.pow(0.9995, this._densTicks);
    this._densTicks = 0;
    const out = this.densData;
    for (let i = 0; i < d.length; i++) {
      d[i] *= kd; tr[i] *= kt;
      out[i * 2] = d[i] * 6 > 255 ? 255 : d[i] * 6; out[i * 2 + 1] = tr[i] * 3 > 255 ? 255 : tr[i] * 3;
    }
    this.densTex.needsUpdate = true;
  }

  dispose() {
    this.scene.remove(this.group);
    this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
    this.texLin.dispose(); this.texNear.dispose(); this.densTex.dispose();
  }
}
