import * as THREE from 'three';

// terrain classes (sim/constants.py)
export const T = { DEEP: 0, SHALLOW: 1, SAND: 2, GRASS: 3, MEADOW: 4, FOREST: 5, BARREN: 6, ROCK: 7, MOUNTAIN: 8, CAVE: 9 };
export const TERRAIN_HEX = [0x2c4a52, 0x6b7a58, 0xdcc892, 0x7fb04a, 0x62b044, 0x3a6e30, 0xab9369, 0x857c70, 0x6b645c, 0x4b443e];
export const WEATHER = ['Clear', 'Rain', 'Fog'];
export const SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter'];
export const SEASON_COLORS = ['#9be58b', '#ffd36b', '#ff9a52', '#a9d8ff'];
export const DEATH_COLORS = { starvation: '#e0a050', thirst: '#5fb6ff', cold: '#b8e3ff', fire: '#ff6a3d',
  predator: '#ff4f6d', old_age: '#c9b8ff', strike: '#ff9de2', other: '#9aa5b5' };
export const ACTION_LABEL = {
  noop: ['·', 'idle'], move_n: ['↑', 'walking'], move_e: ['→', 'walking'], move_s: ['↓', 'walking'], move_w: ['←', 'walking'],
  rest: ['☾', 'resting'], eat: ['🍃', 'eating'], drink: ['💧', 'drinking'], hunt: ['⚔', 'hunting'],
  pickup: ['✋', 'picking up'], drop: ['⇣', 'dropping'], build: ['⌂', 'building'], strike: ['✊', 'striking'],
};

// deterministic PRNG for decoration placement
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function hash2(x, y) {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
export function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
export function agentColor(hue, tone) {
  const c = new THREE.Color();
  c.setHSL(hue / 255, 0.55 + tone / 15 * 0.25, 0.5);
  return c;
}
export function skinColor(tone) {
  const c = new THREE.Color();
  c.setHSL(0.07, 0.45, 0.32 + tone / 15 * 0.38);
  return c;
}

// GLSL helpers injected into materials
export const GLSL_NOISE = /* glsl */`
float gHash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float gNoise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f);
  return mix(mix(gHash(i),gHash(i+vec2(1.,0.)),u.x), mix(gHash(i+vec2(0.,1.)),gHash(i+vec2(1.,1.)),u.x), u.y); }
float gFbm(vec2 p){ float v=0., a=.5; for(int i=0;i<4;i++){ v+=a*gNoise(p); p*=2.03; a*=.5; } return v; }
float gFlag(float a, float bit){ return mod(floor(floor(a*255.+.5)/bit), 2.); }
`;

export function toast(msg, ms = 2600) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove('show'), ms);
}
