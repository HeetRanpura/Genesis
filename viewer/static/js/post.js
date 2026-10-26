// Post-processing: MSAA scene -> bloom -> tone mapping (OutputPass) -> grade pass with tilt-shift,
// split-tone colour grade, vignette, chromatic fringe and film grain.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';

const GradeShader = {
  uniforms: { tDiffuse: { value: null }, uRes: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 }, uTilt: { value: 0.6 },
    uNight: { value: 0 }, uGrain: { value: 0.045 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse; uniform vec2 uRes; uniform float uTime, uTilt, uNight, uGrain; varying vec2 vUv;
    float rnd(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + uTime * 0.0) * 43758.5453); }
    void main(){
      vec2 uv = vUv;
      float band = smoothstep(0.18, 0.62, abs(uv.y - 0.52)) * uTilt;
      vec3 col = texture2D(tDiffuse, uv).rgb; float wsum = 1.0;
      vec2 px = 1.0 / uRes;
      if (band > 0.02) for (int i = 1; i < 7; i++) {
        float a = float(i) * 2.39996; float r = sqrt(float(i) / 6.0);
        vec2 o = vec2(cos(a), sin(a)) * r * band * 4.5 * px * (uRes.y / 1080.0) ;
        float w = 1.0 - r * 0.5;
        col += texture2D(tDiffuse, uv + o).rgb * w; wsum += w;
      }
      col /= wsum;
      // chromatic fringe toward edges
      vec2 dc = uv - 0.5;
      float ca = dot(dc, dc) * 0.0035;
      col.r = mix(col.r, texture2D(tDiffuse, uv + dc * ca * 4.0).r, 0.6);
      col.b = mix(col.b, texture2D(tDiffuse, uv - dc * ca * 4.0).b, 0.6);
      // split-tone grade: warm highlights, cool shadows (stronger at night)
      float l = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(col, col * vec3(1.06, 1.0, 0.92), smoothstep(0.35, 0.9, l) * 0.6);
      col = mix(col, col * vec3(0.9, 0.97, 1.12), (1.0 - smoothstep(0.0, 0.35, l)) * (0.4 + 0.4 * uNight));
      col = mix(vec3(l), col, 1.08);
      col = (col - 0.5) * 1.04 + 0.5;
      // vignette
      float v = smoothstep(0.95, 0.25, length(dc * vec2(uRes.x / uRes.y, 1.0) * 0.9));
      col *= mix(0.62, 1.0, v);
      // grain
      float g = fract(sin(dot(uv * uRes + fract(uTime) * 100.0, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
      col += g * uGrain;
      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }`,
};

export class Post {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.rt = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
    const rt = this.rt;
    this.composer = new EffectComposer(renderer, rt);
    this.composer.addPass(new RenderPass(scene, camera));
    // ground-truth ambient occlusion: contact shadows under bodies, in crevices and forests
    this.gtao = new GTAOPass(scene, camera, size.x, size.y);
    this.gtao.blendIntensity = 0.85;
    this.gtao.updateGtaoMaterial({ radius: 0.6, distanceExponent: 1.5, thickness: 1.0, scale: 1.0, samples: 12 });
    this.composer.addPass(this.gtao);
    this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x / 4, size.y / 4), 0.35, 0.55, 0.9);   // quarter-res
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.grade = new ShaderPass(GradeShader);
    this.composer.addPass(this.grade);
    this.enabled = true;
  }
  setAO(on) { this.gtao.enabled = on; }
  setQuality(hq) {
    this.hq = hq;
    const n = hq ? 4 : 2;
    for (const t of [this.composer.renderTarget1, this.composer.renderTarget2]) {
      if (t.samples !== n) { t.samples = n; t.dispose(); }   // recreated at the new sample count on next use
    }
  }
  setSize(w, h, pr) {
    this.composer.setPixelRatio(pr);
    this.composer.setSize(w, h);
    this.grade.uniforms.uRes.value.set(w * pr, h * pr);
  }
  render(t, night, tilt, fire = false) {
    // bloom only where it shows (night glow, fire, HQ); it is a full-screen cost otherwise
    this.bloom.enabled = this.hq || night > 0.15 || fire;
    this.grade.uniforms.uTime.value = t;
    this.grade.uniforms.uNight.value = night;
    this.grade.uniforms.uTilt.value = tilt;
    this.bloom.strength = 0.28 + night * 0.55;
    this.bloom.threshold = 0.92 - night * 0.25;
    this.composer.render();
  }
}
