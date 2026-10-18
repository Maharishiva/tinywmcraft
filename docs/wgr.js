// wgr.js: runs one static network converted by convert.py on WebGPU, in plain JS and WGSL (no wasm).
// create() makes every GPU object once: one arena buffer for all tensors, the weights (fp16 ones stay fp16 and kernels read
// them with unpack2x16float; a convert_web.py --int8 file also has int8 ones, 4 to a u32, that kernels multiply by fp32
// scales), one uniform buffer with each dispatch's tensor offsets, ~75 pipelines and a single bind group. run() writes
// the inputs, records the ~1100 dispatches into one compute pass, copies the outputs out and maps them: per step one
// command encoder, one pass, one command buffer, nothing else.

const EXPAND = `@group(0) @binding(0) var<storage, read> R: array<u32>;
@group(0) @binding(1) var<storage, read_write> W: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.x + g.y * nw.x * 256u; if (i >= arrayLength(&R)) { return; }
  let v = unpack2x16float(R[i]); W[2u * i] = v.x; W[2u * i + 1u] = v.y;
}`;

export function parse(bytes) {  // 'WGR1', u32 json length, json (padded to 4 bytes), fp16 weights, fp32 weights, int8 weights
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (String.fromCharCode(...u8.subarray(0, 4)) !== 'WGR1') throw new Error('not a wgr file');
  const n = new DataView(u8.buffer, u8.byteOffset + 4, 4).getUint32(0, true);
  return { meta: JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n))), weights: u8.subarray(8 + n) };
}

export async function create(device, bytes, { chunk = 1 << 22 } = {}) {
  const { meta: m, weights } = parse(bytes);
  const U = GPUBufferUsage, C = GPUShaderStage.COMPUTE;
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  const A = device.createBuffer({ size: m.arena * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // weights: fp16 ones in H as they are (an --f32 file has none: they are expanded into W here, chunk by chunk)
  const b16 = m.n16 * 2, wbase = m.f16 ? 0 : m.n16;
  const W = device.createBuffer({ size: Math.max(16, (wbase + m.n32) * 4), usage: U.STORAGE | U.COPY_DST });
  const H = m.f16 ? device.createBuffer({ size: Math.max(16, b16), usage: U.STORAGE | U.COPY_DST }) : null;
  const n8 = m.n8 ?? 0, Q = n8 ? device.createBuffer({ size: n8, usage: U.STORAGE | U.COPY_DST }) : null;
  if (m.n32) device.queue.writeBuffer(W, wbase * 4, weights, b16, m.n32 * 4);
  if (Q) device.queue.writeBuffer(Q, 0, weights, b16 + m.n32 * 4, n8);   // n8: bytes, a multiple of 4
  if (H) device.queue.writeBuffer(H, 0, weights, 0, b16);
  else if (b16) {
    const ex = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: EXPAND }), entryPoint: 'main' } });
    const step = Math.min(chunk, b16), raw = device.createBuffer({ size: step, usage: U.STORAGE | U.COPY_DST });
    for (let at = 0; at < b16; at += step) {
      const len = Math.min(step, b16 - at);   // bytes of fp16, a multiple of 4 since convert.py pads n16 to even
      device.queue.writeBuffer(raw, 0, weights, at, len);
      const bg = device.createBindGroup({ layout: ex.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: raw, offset: 0, size: len } },
        { binding: 1, resource: { buffer: W, offset: at * 2, size: len * 2 } }] });
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass(), wg = Math.ceil(len / 4 / 256);
      pass.setPipeline(ex);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.min(wg, 65535), Math.ceil(wg / 65535));
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    raw.destroy();
  }

  // every dispatch's tensor offsets, 256 bytes each (the dynamic uniform offset alignment)
  const n = m.dispatches.length, prm = new Uint32Array(n * 64);
  m.params.forEach((p, i) => prm.set(p, i * 64));
  const P = device.createBuffer({ size: prm.byteLength, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(P, 0, prm);
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: C, buffer: { type: 'storage' } },
    { binding: 2, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 256 } },
    ...(H ? [{ binding: 3, visibility: C, buffer: { type: 'read-only-storage' } }] : []),
    ...(Q ? [{ binding: 4, visibility: C, buffer: { type: 'read-only-storage' } }] : [])] });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipes = await Promise.all(m.shaders.map(s => device.createComputePipelineAsync({ layout,
    compute: { module: device.createShaderModule({ code: m.header + s }), entryPoint: 'main' } })));
  const bg = device.createBindGroup({ layout: bgl, entries: [
    { binding: 0, resource: { buffer: W } }, { binding: 1, resource: { buffer: A } },
    { binding: 2, resource: { buffer: P, offset: 0, size: 256 } }, ...(H ? [{ binding: 3, resource: { buffer: H } }] : []),
    ...(Q ? [{ binding: 4, resource: { buffer: Q } }] : [])] });
  const dyn = new Uint32Array(n).map((_, i) => i * 256);
  const disp = m.dispatches.map(([s, x, y, z]) => [pipes[s], x, y, z]);
  const staging = new Float32Array(m.in_words);
  const read = device.createBuffer({ size: m.out_words * 4, usage: U.MAP_READ | U.COPY_DST });
  const oom = await device.popErrorScope(), err = await device.popErrorScope();
  if (oom || err) throw new Error((oom || err).message);

  async function step(x) {
    for (const [k, v] of Object.entries(m.inputs)) staging.set(x[k], v.offset);   // int32 inputs become floats
    device.queue.writeBuffer(A, 0, staging);
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    let last = null;
    for (let i = 0; i < n; i++) {
      const [p, gx, gy, gz] = disp[i];
      if (p !== last) pass.setPipeline(last = p);
      pass.setBindGroup(0, bg, dyn, i, 1);
      pass.dispatchWorkgroups(gx, gy, gz);
    }
    pass.end();
    enc.copyBufferToBuffer(A, m.out_start * 4, read, 0, m.out_words * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const b = read.getMappedRange().slice(0);
    read.unmap();
    const out = {};
    for (const [k, v] of Object.entries(m.outputs)) {
      const at = (v.offset - m.out_start) * 4, len = v.shape.reduce((a, b) => a * b, 1);
      out[k] = v.dtype === 'uint8' ? new Uint8Array(b, at, len) : new Float32Array(b, at, len);
    }
    return out;
  }
  let queue = Promise.resolve();   // calls run one after another (they share the arena and the read buffer)
  const run = x => (queue = queue.then(() => step(x), () => step(x)));   // x: the ONNX inputs as typed arrays
  return { run, meta: m, dispatches: n, pipelines: pipes.length, arenaBytes: A.size,
    weightBytes: W.size + (H?.size ?? 0) + (Q?.size ?? 0),
    destroy() { A.destroy(); W.destroy(); H?.destroy(); Q?.destroy(); P.destroy(); read.destroy(); } };
}
