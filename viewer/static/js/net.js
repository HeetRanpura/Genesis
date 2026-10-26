// WebSocket client + decoder for the server's binary messages:
// [u32 header_len][header JSON][buffers aligned to 4 bytes], buffer table in header.bufs.
const TYPES = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, i32: Int32Array, f32: Float32Array };
const td = new TextDecoder();

export function decode(buf) {
  const dv = new DataView(buf);
  const hl = dv.getUint32(0, true);
  const hdr = JSON.parse(td.decode(new Uint8Array(buf, 4, hl)));
  const base = 4 + hl;
  hdr.data = {};
  for (const b of hdr.bufs) hdr.data[b.name] = new TYPES[b.dtype](buf, base + b.offset, b.length);
  return hdr;
}

export class Net {
  constructor(handlers) {
    this.h = handlers;
    this.connected = false;
    this.bytes = 0;
    this.connect();
  }
  connect() {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { this.connected = true; this.h.open?.(); };
    ws.onclose = () => { this.connected = false; this.h.close?.(); setTimeout(() => this.connect(), 1500); };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') { this.h.message?.(JSON.parse(e.data)); return; }
      this.bytes += e.data.byteLength;
      this.h.message?.(decode(e.data));
    };
    this.ws = ws;
  }
  send(cmd) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(cmd)); }
}
