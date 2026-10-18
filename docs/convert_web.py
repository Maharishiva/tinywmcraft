"""play.onnx (from export_web.py) -> play.wgr, what the browser player runs with wgr.js: kernels, dispatches, memory plan and
the weights (fp16 where the ONNX stores fp16).

    python docs/convert_web.py [--onnx docs/play.onnx] [--out docs/play.wgr]
    python docs/convert_web.py --int8 [--group 64] --out docs/play_int8.wgr     # int8 matmul and conv weights, 12 MB

The JS runtime (wgr.js) only creates buffers, pipelines and one bind group, then replays the dispatch list. Everything else
is decided here, for this one static graph:
- constant folding of the Casts in front of the fp16 initializers (those weights stay fp16 in the blob, the GPU expands them),
- Reshape/Squeeze/Cast-to-float are views (no copy), every tensor lives in one f32 arena (ints and bools as floats),
- elementwise ops, Transpose, Slice, Split, Concat, Expand and Gather are inlined as WGSL functions of the output index
  into whatever reads them (no dispatch of their own unless their result is read twice),
- Gemm, Einsum, Conv, ConvTranspose take the elementwise tree that consumes them as an epilogue,
- Softmax, ReduceMean, ReduceSum read their input through the same inlined functions,
- a liveness-based plan packs every materialized tensor into the arena.
--int8 stores the weights of Gemm, Einsum, Conv and ConvTranspose (16384 elements and up) as symmetric int8 with an fp32
scale per output channel (or per --group inputs), 4 to a u32; kernels compute q * scale in fp32. Embeddings, norms,
biases and position tables stay fp16. Constants folded from such a weight are folded again from its int8 values, so the
file computes exactly what the model with dequantized weights computes.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto

DOCS = Path(__file__).resolve().parent

BIN = {'Add': '+', 'Sub': '-', 'Mul': '*', 'Div': '/'}
UN = {'Sqrt', 'Tanh', 'Cos', 'Sin', 'Exp', 'Sigmoid', 'Abs', 'Neg', 'Relu'}
EW = set(BIN) | UN | {'Less', 'Greater', 'Where', 'Clip', 'Cast', 'Expand'}
REMAP = {'Transpose', 'Slice', 'Split', 'Concat', 'Gather'}
HEAVY = {'Gemm', 'Einsum', 'Softmax', 'ReduceMean', 'ReduceSum', 'Conv', 'ConvTranspose'}
MATLIKE = {'Gemm', 'Einsum', 'Conv', 'ConvTranspose'}
VIEW = {'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity'}
NDATA = {'Slice': 1, 'Split': 1, 'ReduceMean': 1, 'ReduceSum': 1, 'Expand': 1, 'Reshape': 1, 'Squeeze': 1, 'Unsqueeze': 1}
SLOTS = 64          # u32 offsets per dispatch: 16 vec4 = 256 bytes, the dynamic uniform offset alignment
ALIGN = 16          # arena allocations in words (64 bytes)

HEADER = """@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read_write> A: array<u32>;
struct Prm { o: array<vec4<u32>, 16> }
@group(0) @binding(2) var<uniform> P: Prm;
var<private> ACC: f32;
"""
H16 = '@group(0) @binding(3) var<storage, read> H: array<u32>;\n'
Q8 = '@group(0) @binding(4) var<storage, read> Q: array<u32>;\n'
OPT = {'f16': True,   # keep the fp16 weights fp16 on the GPU (read with unpack2x16float), instead of expanding them to fp32
       'group': 0}    # --int8 --group: inputs per scale of the int8 weights (0: one scale per output channel)
QMAX = 127


def quantize(c, group=0, qmax=QMAX):
    """Symmetric integer weights for c (taps, inputs, outputs), computed in fp32: one scale per output channel (group 0)
    or per group of `group` consecutive inputs of one tap and one output; scale = absmax / qmax, q = round(w / scale) in
    [-qmax, qmax]. Returns q (int8, the shape of c) and the scales: (outputs,) or (taps, inputs // group, outputs).
    The browser computes q * scale in fp32 (exact integer, one correctly rounded product), so q.astype(f32) * scale is
    the weight it uses, bit for bit."""
    c = np.asarray(c, np.float32)
    T, I, O = c.shape
    if group:
        group = min(group, I)
        assert I % group == 0, (c.shape, group)
        g = c.reshape(T, I // group, group, O)
        scale = (np.abs(g).max(2) / np.float32(qmax)).astype(np.float32)
        scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
        q = np.clip(np.rint(g / scale[:, :, None]), -qmax, qmax).astype(np.int8).reshape(T, I, O)
    else:
        scale = (np.abs(c).max((0, 1)) / np.float32(qmax)).astype(np.float32)
        scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
        q = np.clip(np.rint(c / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def dequantize(q, scale):
    return q.astype(np.float32) * (scale if scale.ndim == 1 else np.repeat(scale, q.shape[1] // scale.shape[1], 1))


# ONNX weight layouts <-> quantize()'s (taps, inputs, outputs): Gemm and Einsum 'abcd,de->abce' (K, N), Conv (O, I, KH,
# KW), ConvTranspose (I, O, KH, KW)
TO_C = {'mm': lambda w: w[None],
        'conv': lambda w: w.transpose(2, 3, 1, 0).reshape(-1, w.shape[1], w.shape[0]),
        'convT': lambda w: w.transpose(2, 3, 0, 1).reshape(-1, w.shape[0], w.shape[1])}
FROM_C = {'mm': lambda c, s: c[0],
          'conv': lambda c, s: c.reshape(s[2], s[3], s[1], s[0]).transpose(3, 2, 0, 1),
          'convT': lambda c, s: c.reshape(s[2], s[3], s[0], s[1]).transpose(2, 3, 0, 1)}


def scale_index(kind, shape, group, i='i'):
    """WGSL index into the scales (stored as quantize() returns them, flat) of element i of an ONNX weight."""
    if kind == 'mm':
        K, N = shape
        g = min(group, K)
        return f'{i} % {N}u' if not group else f'({i} / {N * g}u) * {N}u + {i} % {N}u'
    if kind == 'conv':
        O, I, KH, KW = shape
        o, c, tap = f'({i} / {I * KH * KW}u)', f'(({i} / {KH * KW}u) % {I}u)', f'({i} % {KH * KW}u)'
    else:
        I, O, KH, KW = shape
        o, c, tap = f'(({i} / {KH * KW}u) % {O}u)', f'({i} / {O * KH * KW}u)', f'({i} % {KH * KW}u)'
    if not group:
        return o
    g = min(group, I)
    return f'({tap} * {I // g}u + {c} / {g}u) * {O}u + {o}'


def prod(s):
    return int(np.prod(s)) if len(s) else 1


def strides(s):
    st = [1] * len(s)
    for d in range(len(s) - 2, -1, -1):
        st[d] = st[d + 1] * s[d + 1]
    return st


def lit(x):
    x = np.float32(x)
    if np.isfinite(x) and float(x) == int(x) and abs(x) < 1e6:
        return f'{int(x)}.0'
    return f'bitcast<f32>({int(x.view(np.uint32))}u)'


def attrs(n):
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def unravel(so, coef, iv='i'):
    """sum over dims d of (index of dim d of the linear index iv in shape so) * coef[d], as a u32 WGSL expression."""
    sos, terms = strides(so), []
    for d in range(len(so)):
        if so[d] == 1 or coef[d] == 0:
            continue
        e = iv if sos[d] == 1 else f'({iv} / {sos[d]}u)'
        if d > 0:
            e = f'({e} % {so[d]}u)'
        if coef[d] != 1:
            e = f'{e} * {coef[d]}u'
        terms.append(e)
    return ' + '.join(terms) or '0u'


def bidx(so, si, iv='i'):
    """index into a tensor of shape si broadcast (numpy rules) to shape so."""
    so, si = tuple(so), tuple(si)
    if si == so or prod(si) == prod(so) and [d for d in si if d != 1] == [d for d in so if d != 1]:
        return iv
    if prod(si) == 1:
        return '0u'
    pad = (1,) * (len(so) - len(si)) + si
    st = strides(pad)
    return unravel(so, [0 if pad[d] == 1 else st[d] for d in range(len(so))], iv)


class Graph:
    def __init__(self, path):
        m = shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
        g = m.graph
        self.shape, self.dtype = {}, {}
        for vi in [*g.value_info, *g.input, *g.output]:
            t = vi.type.tensor_type
            self.shape[vi.name] = tuple(d.dim_value for d in t.shape.dim)
            self.dtype[vi.name] = t.elem_type
        self.const = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        for k, v in self.const.items():
            self.shape[k] = tuple(v.shape)
        self.f16, self.q8 = set(), {}
        self.inputs = [i.name for i in g.input if i.name not in self.const]
        self.outputs = [o.name for o in g.output]
        self.nodes, self.folded = [], []
        for n in g.node:
            if all(x == '' or x in self.const for x in n.input):
                self.fold(n)
                self.folded.append(n)
            else:
                self.nodes.append(n)
        self.base = {}
        for n in self.nodes:
            if self.is_view(n):
                self.base[n.output[0]] = self.B(n.input[0])
        self.prod = {}
        for n in self.nodes:
            if not self.is_view(n):
                for o in n.output:
                    self.prod[o] = n
        self.uses = defaultdict(int)
        for n in self.nodes:
            if not self.is_view(n):
                for x in self.data(n):
                    self.uses[self.B(x)] += 1
        for o in self.outputs:
            self.uses[self.B(o)] += 1

    def fold(self, n):
        a, x = attrs(n), [self.const.get(i) for i in n.input]
        if n.op_type == 'Cast':
            v = x[0].astype(helper.tensor_dtype_to_np_dtype(a['to']))
            if x[0].dtype == np.float16 and a['to'] == TensorProto.FLOAT:
                self.f16.add(n.output[0])
        elif n.op_type in ('Reshape', 'Identity'):
            v = x[0].reshape(x[1]) if n.op_type == 'Reshape' else x[0]
        else:
            from onnx.reference import ReferenceEvaluator
            sub = helper.make_graph([n], 'f', [], [helper.make_tensor_value_info(o, 0, None) for o in n.output],
                                    [numpy_helper.from_array(self.const[i], i) for i in n.input if i])
            v = ReferenceEvaluator(helper.make_model(sub, opset_imports=[helper.make_opsetid('', 18)])).run(None, {})[0]
        self.const[n.output[0]] = np.asarray(v)
        self.shape[n.output[0]] = tuple(np.shape(v))

    def is_view(self, n):
        return n.op_type in VIEW or n.op_type == 'Cast' and attrs(n)['to'] == TensorProto.FLOAT

    def B(self, t):
        return self.base.get(t, t)

    def data(self, n):
        k = NDATA.get(n.op_type)
        return [x for x in (n.input[:k] if k else n.input) if x]

    def cval(self, t):
        return self.const[self.B(t)]

    def mark_int8(self, group=0, min_size=16384):
        """--int8: every fp16 weight with at least min_size elements that is only read as the weight of a Gemm, an Einsum
        'abcd,de->abce', a Conv or a ConvTranspose becomes int8 with fp32 scales (quantize()); self.q8: name -> (kind,
        shape, q in the ONNX layout, flat scales). Embeddings, norms, biases and position tables stay fp16."""
        reads = defaultdict(set)
        for n in self.nodes:
            for j, x in enumerate(n.input):
                b = self.B(x) if x else None
                if b not in self.f16:
                    continue
                kind = None
                if j == 1 and self.shape[x] == self.shape[b] and attrs(n).get('group', 1) == 1:
                    eq = attrs(n).get('equation', b'')
                    eq = eq.decode() if isinstance(eq, bytes) else eq
                    kind = {'Gemm': 'mm', 'Conv': 'conv', 'ConvTranspose': 'convT'}.get(n.op_type)
                    if n.op_type == 'Einsum' and eq == 'abcd,de->abce':
                        kind = 'mm'
                    if n.op_type == 'Gemm' and (attrs(n).get('transA', 0) or attrs(n).get('transB', 0)):
                        kind = None
                reads[b].add(kind)
        self.q8 = {}
        for t, kinds in reads.items():
            if len(kinds) == 1 and None not in kinds and prod(self.shape[t]) >= min_size:
                kind, w = next(iter(kinds)), self.const[t].astype(np.float32)
                q, s = quantize(TO_C[kind](w), group)
                self.q8[t] = (kind, w.shape, FROM_C[kind](q, w.shape), s)
        # constants folded from an int8 weight (the conditioning of each fixed noise level reads taulin's weight) are
        # folded again from its dequantized values: the file then computes exactly what the dequantized model computes
        for t, (kind, shape, q, s) in self.q8.items():
            self.const[t] = FROM_C[kind](dequantize(TO_C[kind](q), s), shape)
        for n in self.folded:
            if not any(o in self.q8 for o in n.output):
                self.fold(n)
        return self.q8


class Plan:
    """Which tensors get a buffer, which heavy op absorbs which elementwise tree, kernel sources and the dispatch list."""

    def __init__(self, G):
        self.G = G
        outs = {G.B(o) for o in G.outputs}
        self.outs = outs
        self.inline = set()
        for t, n in G.prod.items():
            op = n.op_type
            if (op in EW or op in REMAP) and G.uses[t] == 1 and t not in outs and not self.is_u8(n):
                self.inline.add(t)
        # an elementwise tree read by a heavy op is computed once, in the epilogue of the heavy op inside it, if any
        for n in G.nodes:
            if n.op_type in HEAVY:
                for x in G.data(n):
                    t = G.B(x)
                    if t in self.inline and self.find_epi(t):
                        self.inline.discard(t)
        self.fused = {}         # root tensor -> heavy tensor computed in its kernel
        for n in G.nodes:
            if G.is_view(n) or n.op_type in HEAVY:
                continue
            for t in n.output:
                if t in self.inline or not t or not G.uses[t] or self.is_u8(n):
                    continue
                h = self.find_epi(t)
                if h:
                    self.fused[t] = h
        self.absorbed = set(self.fused.values())

    def is_u8(self, n):
        return n.op_type == 'Cast' and attrs(n)['to'] == TensorProto.UINT8

    def find_epi(self, R):
        """a Gemm/Einsum/Conv output read only by R's inlined tree, at R's own linear index."""
        G, hits, bad = self.G, defaultdict(int), set()

        def walk(t, aligned):
            n = G.prod[t]
            for x in G.data(n):
                bx = G.B(x)
                al = aligned and n.op_type in EW and n.op_type != 'Expand' and G.shape[x] == G.shape[n.output[0]]
                if bx in self.inline and bx != R:
                    walk(bx, al)
                elif bx in G.prod and G.prod[bx].op_type in MATLIKE and bx not in self.outs:
                    if al:
                        hits[bx] += 1
                    else:
                        bad.add(bx)
        walk(R, True)
        for h, c in hits.items():
            if h not in bad and c == G.uses[h] and prod(G.shape[h]) == prod(G.shape[R]):
                return h
        return None


class Kernel:
    """WGSL for one dispatch: functions f(i) giving each inlined tensor at linear index i, and a main."""

    def __init__(self, plan, root, acc=None):
        self.P, self.G, self.root, self.acc = plan, plan.G, root, acc
        self.fns, self.lines, self.slots, self.slotmap = {}, [], [], {}

    def slot(self, key):
        if key not in self.slotmap:
            self.slotmap[key] = len(self.slots)
            self.slots.append(key)
        k = self.slotmap[key]
        assert k < SLOTS
        return f'P.o[{k // 4}].{"xyzw"[k % 4]}'

    def call(self, x, idx):
        G, bx = self.G, self.G.B(x)
        if bx in G.const and prod(G.shape[bx]) == 1 and bx not in G.f16:
            return lit(G.const[bx].ravel()[0])
        return f'{self.f(bx)}({idx})'

    def f(self, t):
        t = self.G.B(t)
        if t in self.fns:
            return self.fns[t]
        name = self.fns[t] = f'f{len(self.fns)}'
        G = self.G
        if t == self.acc:
            body = 'return ACC;'
        elif t != self.root and (t in G.const):
            if prod(G.shape[t]) == 1:
                body = f'return {lit(G.const[t].ravel()[0])};'
            elif t in G.q8:      # int8, 4 to a u32 (little-endian), sign-extended by shifts, times its fp32 scale
                kind, shape = G.q8[t][:2]
                body = (f'let j = {self.slot(("Q", t))} + i; '
                        f'let q = f32((bitcast<i32>(Q[j >> 2u]) << ((3u - (j & 3u)) << 3u)) >> 24u); '
                        f'return q * W[{self.slot(("S", t))} + {scale_index(kind, shape, OPT["group"])}];')
            elif OPT['f16'] and t in G.f16:
                body = (f'let j = {self.slot(("H", t))} + i; let v = unpack2x16float(H[j >> 1u]); '
                        f'return select(v.x, v.y, (j & 1u) == 1u);')
            else:
                body = f'return W[{self.slot(("W", t))} + i];'
        elif t != self.root and t not in self.P.inline:
            body = f'return bitcast<f32>(A[{self.slot(("A", t))} + i]);'
        else:
            body = self.node_body(G.prod[t], t)
        self.lines.append(f'fn {name}(i: u32) -> f32 {{ {body} }}')
        return name

    def node_body(self, n, t):
        G, op, so = self.G, n.op_type, self.G.shape[t]
        ins, a = G.data(n), attrs(n)
        if op in BIN:
            x, y = ins
            return f'return {self.call(x, bidx(so, G.shape[x]))} {BIN[op]} {self.call(y, bidx(so, G.shape[y]))};'
        if op in ('Less', 'Greater'):
            x, y = ins
            c = '<' if op == 'Less' else '>'
            return f'return select(0.0, 1.0, {self.call(x, bidx(so, G.shape[x]))} {c} {self.call(y, bidx(so, G.shape[y]))});'
        if op in UN:
            v = self.call(ins[0], bidx(so, G.shape[ins[0]]))
            return 'return ' + {'Sqrt': f'sqrt({v})', 'Tanh': f'tanh(clamp({v}, -10.0, 10.0))', 'Cos': f'cos({v})',
                                'Sin': f'sin({v})', 'Exp': f'exp({v})', 'Sigmoid': f'1.0 / (1.0 + exp(-({v})))',
                                'Abs': f'abs({v})', 'Neg': f'-({v})', 'Relu': f'max({v}, 0.0)'}[op] + ';'
        if op == 'Cast':
            v = self.call(ins[0], bidx(so, G.shape[ins[0]]))
            to = a['to']
            if to in (TensorProto.INT32, TensorProto.INT64, TensorProto.INT8, TensorProto.INT16):
                return f'return trunc({v});'
            if to == TensorProto.BOOL:
                return f'return select(0.0, 1.0, {v} != 0.0);'
            raise NotImplementedError(f'Cast to {to}')
        if op == 'Where':
            c, x, y = ins
            return (f'return select({self.call(y, bidx(so, G.shape[y]))}, {self.call(x, bidx(so, G.shape[x]))}, '
                    f'{self.call(c, bidx(so, G.shape[c]))} != 0.0);')
        if op == 'Clip':
            v = self.call(ins[0], 'i')
            lo = n.input[1] if len(n.input) > 1 and n.input[1] else None
            hi = n.input[2] if len(n.input) > 2 and n.input[2] else None
            if lo:
                v = f'max({v}, {self.call(lo, "0u")})'
            if hi:
                v = f'min({v}, {self.call(hi, "0u")})'
            return f'return {v};'
        if op == 'Expand':
            return f'return {self.call(ins[0], bidx(so, G.shape[ins[0]]))};'
        if op == 'Transpose':
            si = G.shape[ins[0]]
            perm = a.get('perm', list(range(len(si)))[::-1])
            sis = strides(si)
            return f'return {self.call(ins[0], unravel(so, [sis[perm[d]] for d in range(len(so))]))};'
        if op == 'Slice':
            si = G.shape[ins[0]]
            starts, ends = G.cval(n.input[1]), G.cval(n.input[2])
            axes = G.cval(n.input[3]) if len(n.input) > 3 and n.input[3] else range(len(starts))
            steps = G.cval(n.input[4]) if len(n.input) > 4 and n.input[4] else [1] * len(starts)
            st, step = [0] * len(si), [1] * len(si)
            for s, ax, sp in zip(starts, axes, steps):
                ax = int(ax) % len(si)
                s = int(s)
                s = s + si[ax] if s < 0 else s
                st[ax], step[ax] = min(max(s, 0), si[ax] - (1 if sp > 0 else 0)), int(sp)
            sis = strides(si)
            off = sum(st[d] * sis[d] for d in range(len(si)))
            return f'return {self.call(ins[0], unravel(so, [step[d] * sis[d] for d in range(len(si))]) + f" + {off}u")};'
        if op == 'Split':
            si = G.shape[ins[0]]
            ax = a.get('axis', 0) % len(si)
            k = list(n.output).index(t)
            sizes = G.cval(n.input[1]) if len(n.input) > 1 and n.input[1] else [si[ax] // len(n.output)] * len(n.output)
            sis = strides(si)
            off = int(sum(sizes[:k])) * sis[ax]
            return f'return {self.call(ins[0], unravel(so, sis) + f" + {off}u")};'
        if op == 'Concat':
            ax = a['axis'] % len(so)
            sos = strides(so)
            o = 'i' if sos[ax] == 1 else f'(i / {sos[ax]}u)'
            if ax > 0:
                o = f'({o} % {so[ax]}u)'
            code, off = [f'let c = {o};'], 0
            for j, x in enumerate(ins):
                sx = G.shape[x]
                sxs = strides(sx)
                idx = unravel(so, sxs) + (f' - {off * sxs[ax]}u' if off else '')
                call = self.call(x, idx)
                if j < len(ins) - 1:
                    code.append(f'if (c < {off + sx[ax]}u) {{ return {call}; }}')
                else:
                    code.append(f'return {call};')
                off += sx[ax]
            return ' '.join(code)
        if op == 'Gather':
            assert a.get('axis', 0) == 0
            d, ix = n.input
            sd = G.shape[d]
            inner = prod(sd[1:])
            j = self.call(ix, 'i' if inner == 1 else f'i / {inner}u')
            r = '' if inner == 1 else f' + i % {inner}u'
            return f'let j = u32({j}); return {self.call(d, f"j * {inner}u{r}")};'
        raise NotImplementedError(op)

    def raw_int8(self, t):
        """With one scale per output channel (--int8, --group 0), the weight operand of a matmul or conv as its integers,
        and a function of the output column giving ' * scale' for the kernel's accumulated sum; (f(t), '') otherwise."""
        t = self.G.B(t)
        if t not in self.G.q8 or OPT['group']:
            return self.f(t), lambda col: ''
        if ('q', t) not in self.fns:
            name = self.fns[('q', t)] = f'f{len(self.fns)}'
            self.lines.append(f'fn {name}(i: u32) -> f32 {{ let j = {self.slot(("Q", t))} + i; '
                              f'return f32((bitcast<i32>(Q[j >> 2u]) << ((3u - (j & 3u)) << 3u)) >> 24u); }}')
        return self.fns[('q', t)], lambda col: f' * W[{self.slot(("S", t))} + {col}]'

    def source(self, main):
        return '\n'.join(self.lines) + '\n' + main


def pow2(x):
    return 1 << max(0, math.ceil(math.log2(max(1, x))))


def mm_desc(G, n):
    """(B1, B2, M, N, K, sa, sb, so) for Gemm and our three Einsum equations: strides over (b1, b2, m, k), (b1, b2, k, n)
    and (b1, b2, m, n) in the inputs' and output's contiguous layouts."""
    a = attrs(n)
    if n.op_type == 'Gemm':
        (M, K), (K2, N) = G.shape[n.input[0]], G.shape[n.input[1]]
        assert not a.get('transA', 0) and not a.get('transB', 0) and K == K2
        return 1, 1, M, N, K, (0, 0, K, 1), (0, 0, N, 1), (0, 0, N, 1)
    eq = a['equation'].decode() if isinstance(a['equation'], bytes) else a['equation']
    sa, sb = G.shape[n.input[0]], G.shape[n.input[1]]
    if eq == 'abcd,de->abce':
        M, K, N = prod(sa[:3]), sa[3], sb[1]
        return 1, 1, M, N, K, (0, 0, K, 1), (0, 0, N, 1), (0, 0, N, 1)
    if eq == '...qhd,...khd->...hqk':
        assert sa[:-3] == sb[:-3]
        E, (Q, H, D), Kn = prod(sa[:-3]), sa[-3:], sb[-3]
        return E, H, Q, Kn, D, (Q * H * D, D, H * D, 1), (Kn * H * D, D, 1, H * D), (H * Q * Kn, Q * Kn, Kn, 1)
    if eq == '...hqk,...khd->...qhd':
        assert sa[:-3] == sb[:-3]
        E, (H, Q, Kk), D = prod(sa[:-3]), sa[-3:], sb[-1]
        return E, H, Q, D, Kk, (H * Q * Kk, Q * Kk, Kk, 1), (Kk * H * D, D, H * D, 1), (Q * H * D, D, H * D, 1)
    raise NotImplementedError(eq)


TILE = {}     # experiments: TILE['small'] = (TN, RN) for 1 < M <= 32, TILE['one'] = TN for M == 1


def tiles(M, N, K):
    """(TM, TN, TK, RM, RN): a 64x64 tile with 4x4 per thread for big matmuls; for the small-M ones (16 tokens of the
    new frame, 1 row of the conditioning) narrow tiles, one or two outputs per thread, so there are enough workgroups."""
    if M == 1:
        TM, TN, RM, RN = 1, min(TILE.get('one', 64), pow2(N)), 1, 1
    elif M <= 32:
        TM = pow2(M)
        TN, RN = TILE.get('small', (16, 1))
        TN = min(TN, pow2(N))
        RM = max(1, TM * TN // (256 * RN))
        RN = min(RN, TN)
    else:
        TM, TN = min(64, pow2(M)), min(64, pow2(N))
        per = max(1, TM * TN // 256)
        RM = 1
        while RM * RM < per and RM < TM:
            RM *= 2
        RN = max(1, per // RM)
    TK = min(TILE.get('tk', 16) if M > 32 else TILE.get('tks', 64), pow2(K))
    return TM, TN, TK, RM, RN


def lin(c, v):
    return ' + '.join(f'{x} * {s}u' if s != 1 else x for x, s in zip(v, c) if s) or '0u'


def build(G):
    P = Plan(G)
    kernels = []        # (source, slots, workgroups, root, reads)

    def store(k, R, val, idx):
        return f'A[{k.slot(("A", R))} + {idx}] = bitcast<u32>({val});'

    def ew_kernel(R):
        n = G.prod[R]
        if P.is_u8(n):
            k = Kernel(P, R)
            fx = k.f(G.B(n.input[0])) if G.B(n.input[0]) in P.inline else k.f(n.input[0])
            cnt = prod(G.shape[R])
            nw = (cnt + 3) // 4
            main = (f'@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n'
                    f'  let w = g.x; if (w >= {nw}u) {{ return; }}\n  var p = 0u;\n'
                    f'  for (var t = 0u; t < 4u; t++) {{ let i = w * 4u + t; if (i < {cnt}u) {{ '
                    f'p = p | (u32(clamp({fx}(i), 0.0, 255.0)) << (8u * t)); }} }}\n'
                    f'  A[{k.slot(("A", R))} + w] = p;\n}}')
            return k, main, ((nw + 127) // 128, 1, 1)
        k = Kernel(P, R, acc=P.fused.get(R))
        fr = k.f(R)
        cnt = prod(G.shape[R])
        main = (f'@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n'
                f'  let i = g.x; if (i >= {cnt}u) {{ return; }}\n  {store(k, R, f"{fr}(i)", "i")}\n}}')
        return k, main, ((cnt + 127) // 128, 1, 1)

    def mm_kernel(R, H):
        n = G.prod[H]
        k = Kernel(P, R, acc=H)
        B1, B2, M, N, K, sa, sb, so = mm_desc(G, n)
        TM, TN, TK, RM, RN = tiles(M, N, K)
        WX, WY = TN // RN, TM // RM
        NT = WX * WY
        la, (lb, sc) = k.f(n.input[0]), k.raw_int8(n.input[1])
        alpha, beta = attrs(n).get('alpha', 1.0), attrs(n).get('beta', 1.0)
        bias = ''
        if n.op_type == 'Gemm' and len(n.input) > 2 and n.input[2] and beta != 0:
            bias = f' + {lit(beta) + " * " if beta != 1 else ""}{k.call(n.input[2], bidx((M, N), G.shape[n.input[2]], "oi"))}'
        fr = k.f(R)
        if M == 1 and K <= 2048:       # the row in workgroup memory: at most 8 KB
            return gemv(k, R, fr, la, lb, bias, alpha, B1, B2, N, K, sa, sb, so, sc)
        mk, kk_, nk = M % TM, K % TK, N % TN     # bounds checks only where a tile can run off the edge

        def loads(total, body):
            out = []
            for j in range((total + NT - 1) // NT):
                e = f'li + {j * NT}u' if j else 'li'
                out.append(f'if ({e} < {total}u) {{ {body(e)} }}' if (j + 1) * NT > total else f'{{ {body(e)} }}')
            return '\n    '.join(out)

        def cond(*c):
            c = [x for x in c if x]
            return ' && '.join(c)

        def load_a(e):
            c = cond(mk and f'm < {M}u', kk_ and f'kk < {K}u')
            v = f'{la}(ba + {lin(sa[2:], ["m", "kk"])})'
            return (f'let e = {e}; let m = m0 + e / {TK}u; let kk = k0 + e % {TK}u; '
                    + (f'var v = 0.0; if ({c}) {{ v = {v}; }} ' if c else f'let v = {v}; ') + f'As[(e % {TK}u) * {TM}u + e / {TK}u] = v;')

        def load_b(e):
            c = cond(kk_ and f'kk < {K}u', nk and f'nn < {N}u')
            v = f'{lb}(bb + {lin(sb[2:], ["kk", "nn"])})'
            return (f'let e = {e}; let kk = k0 + e / {TN}u; let nn = n0 + e % {TN}u; '
                    + (f'var v = 0.0; if ({c}) {{ v = {v}; }} ' if c else f'let v = {v}; ') + 'Bs[e] = v;')

        accs = ' '.join(f'var c{r}_{c}: f32 = 0.0;' for r in range(RM) for c in range(RN))
        av = ' '.join(f'let a{r} = As[kk * {TM}u + lid.y{f" + {r * WY}u" if r else ""}];' for r in range(RM))
        bv = ' '.join(f'let b{c} = Bs[kk * {TN}u + lid.x{f" + {c * WX}u" if c else ""}];' for c in range(RN))
        fm = ' '.join(f'c{r}_{c} = fma(a{r}, b{c}, c{r}_{c});' for r in range(RM) for c in range(RN))
        epi = []
        for r in range(RM):
            for c in range(RN):
                g = cond(mk and f'm < {M}u', nk and f'nn < {N}u')
                body = (f'let oi = bo + {lin(so[2:], ["m", "nn"])}; ACC = {lit(alpha) + " * " if alpha != 1 else ""}c{r}_{c}{sc("nn")}{bias}; '
                        + store(k, R, f'{fr}(oi)', 'oi'))
                epi.append(f'{{ let m = m0 + lid.y{f" + {r * WY}u" if r else ""}; let nn = n0 + lid.x{f" + {c * WX}u" if c else ""}; '
                           + (f'if ({g}) {{ {body} }} }}' if g else f'{body} }}'))
        epi = '\n  '.join(epi)
        main = f"""var<workgroup> As: array<f32, {TK * TM}>;
var<workgroup> Bs: array<f32, {TK * TN}>;
@compute @workgroup_size({WX}, {WY}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
  let b1 = wg.z / {B2}u; let b2 = wg.z % {B2}u;
  let m0 = wg.y * {TM}u; let n0 = wg.x * {TN}u;
  let ba = {lin(sa[:2], ['b1', 'b2'])}; let bb = {lin(sb[:2], ['b1', 'b2'])}; let bo = {lin(so[:2], ['b1', 'b2'])};
  {accs}
  for (var k0 = 0u; k0 < {K}u; k0 += {TK}u) {{
    {loads(TM * TK, load_a)}
    {loads(TK * TN, load_b)}
    workgroupBarrier();
    for (var kk = 0u; kk < {TK}u; kk++) {{
      {av} {bv}
      {fm}
    }}
    workgroupBarrier();
  }}
  {epi}
}}"""
        return k, main, ((N + TN - 1) // TN, (M + TM - 1) // TM, B1 * B2)

    def gemv(k, R, fr, la, lb, bias, alpha, B1, B2, N, K, sa, sb, so, sc):
        """one row: the row in shared memory, K split over KS threads per column, 4 independent sums per thread."""
        TN, KS = TILE.get('gv', (64, 4))
        TN = min(TN, pow2(N))
        KS = min(KS, max(1, 256 // TN))
        NT, U = TN * KS, 4
        st = KS * U
        body = ' '.join(f's{u} = fma(As[k + {u * KS}u], {lb}(bb + (k + {u * KS}u) * {sb[2]}u + n * {sb[3]}u), s{u});' for u in range(U))
        tail = (f'for (; k < {K}u; k += {KS}u) {{ s0 = fma(As[k], {lb}(bb + k * {sb[2]}u + n * {sb[3]}u), s0); }}'
                if K % st else '')
        g = f'n < {N}u' if N % TN else 'true'
        main = f"""var<workgroup> As: array<f32, {K}>;
var<workgroup> red: array<f32, {NT}>;
@compute @workgroup_size({TN}, {KS}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
  let b1 = wg.z / {B2}u; let b2 = wg.z % {B2}u;
  let ba = {lin(sa[:2], ['b1', 'b2'])}; let bb = {lin(sb[:2], ['b1', 'b2'])}; let bo = {lin(so[:2], ['b1', 'b2'])};
  for (var j = li; j < {K}u; j += {NT}u) {{ As[j] = {la}(ba + j * {sa[3]}u); }}
  workgroupBarrier();
  let n = wg.x * {TN}u + lid.x;
  var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
  if ({g}) {{
    var k = lid.y;
    for (; k + {st - KS}u < {K}u; k += {st}u) {{ {body} }}
    {tail}
  }}
  red[li] = (s0 + s1) + (s2 + s3);
  workgroupBarrier();
  if (lid.y == 0u && {g}) {{
    var t = 0.0; for (var j = 0u; j < {KS}u; j++) {{ t += red[j * {TN}u + lid.x]; }}
    let oi = bo + n * {so[3]}u; ACC = {lit(alpha) + " * " if alpha != 1 else ""}t{sc("n")}{bias};
    {store(k, R, f"{fr}(oi)", "oi")}
  }}
}}"""
        return k, main, ((N + TN - 1) // TN, 1, B1 * B2)

    def conv_kernel(R, H):
        n = G.prod[H]
        k = Kernel(P, R, acc=H)
        a = attrs(n)
        x, w = n.input[0], n.input[1]
        (_, Ci, Hi, Wi), so = G.shape[x], G.shape[H]
        _, Co, Ho, Wo = so
        KH, KW = G.shape[w][2:]
        s = a.get('strides', [1, 1])
        assert a.get('group', 1) == 1 and a.get('dilations', [1, 1]) == [1, 1] and so[0] == 1
        fx, (fw, sc) = k.f(x), k.raw_int8(w)
        bias = f' + {k.call(n.input[2], "co")}' if len(n.input) > 2 and n.input[2] else ''
        if n.op_type == 'Conv':
            ap = a.get('auto_pad', b'NOTSET')
            ap = ap.decode() if isinstance(ap, bytes) else ap
            if ap in ('SAME_UPPER', 'SAME_LOWER'):
                pads = []
                for d, (i_, kk) in enumerate([(Hi, KH), (Wi, KW)]):
                    tot = max(0, (math.ceil(i_ / s[d]) - 1) * s[d] + kk - i_)
                    pads.append(tot // 2 if ap == 'SAME_UPPER' else tot - tot // 2)
            else:
                pads = list(a.get('pads', [0, 0, 0, 0]))[:2]
            inner = (f'let iy = i32(oy * {s[0]}u + ky) - {pads[0]}; let ix = i32(ox * {s[1]}u + kx) - {pads[1]};\n'
                     f'      if (iy < 0 || ix < 0 || iy >= {Hi} || ix >= {Wi}) {{ continue; }}\n'
                     f'      acc = fma({fx}((ci * {Hi}u + u32(iy)) * {Wi}u + u32(ix)), {fw}(((co * {Ci}u + ci) * {KH}u + ky) * {KW}u + kx), acc);')
        else:
            assert a.get('auto_pad', b'NOTSET') in (b'NOTSET', 'NOTSET') and not a.get('output_padding')
            pads = list(a.get('pads', [0, 0, 0, 0]))[:2]
            inner = (f'let ty = i32(oy) + {pads[0]} - i32(ky); let tx = i32(ox) + {pads[1]} - i32(kx);\n'
                     f'      if (ty < 0 || tx < 0 || ty % {s[0]} != 0 || tx % {s[1]} != 0) {{ continue; }}\n'
                     f'      let iy = ty / {s[0]}; let ix = tx / {s[1]}; if (iy >= {Hi} || ix >= {Wi}) {{ continue; }}\n'
                     f'      acc = fma({fx}((ci * {Hi}u + u32(iy)) * {Wi}u + u32(ix)), {fw}(((ci * {Co}u + co) * {KH}u + ky) * {KW}u + kx), acc);')
        cnt = prod(so)
        fr = k.f(R)
        main = (f'@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n'
                f'  let oi = g.x; if (oi >= {cnt}u) {{ return; }}\n'
                f'  let co = oi / {Ho * Wo}u; let oy = (oi / {Wo}u) % {Ho}u; let ox = oi % {Wo}u;\n  var acc = 0.0;\n'
                f'  for (var ci = 0u; ci < {Ci}u; ci++) {{ for (var ky = 0u; ky < {KH}u; ky++) {{ for (var kx = 0u; kx < {KW}u; kx++) {{\n'
                f'      {inner}\n  }} }} }}\n  ACC = acc{sc("co")}{bias};\n  {store(k, R, f"{fr}(oi)", "oi")}\n}}')
        return k, main, ((cnt + 63) // 64, 1, 1)

    def red_kernel(R):
        n = G.prod[R]
        k = Kernel(P, R)
        x = n.input[0]
        sx = G.shape[x]
        fx = k.f(x)
        if n.op_type == 'Softmax':
            ax = attrs(n).get('axis', -1) % len(sx)
            assert ax == len(sx) - 1
            L, rows = sx[-1], prod(sx) // sx[-1]
            main = (f'@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n'
                    f'  let r = g.x; if (r >= {rows}u) {{ return; }}\n  let b = r * {L}u;\n'
                    f'  var mx = -3.0e38; for (var j = 0u; j < {L}u; j++) {{ mx = max(mx, {fx}(b + j)); }}\n'
                    f'  var s = 0.0; for (var j = 0u; j < {L}u; j++) {{ s += exp({fx}(b + j) - mx); }}\n'
                    f'  for (var j = 0u; j < {L}u; j++) {{ {store(k, R, f"exp({fx}(b + j) - mx) / s", "b + j")} }}\n}}')
            return k, main, ((rows + 63) // 64, 1, 1)
        axes = [int(v) % len(sx) for v in G.cval(n.input[1])]
        assert len(axes) == 1
        ax = axes[0]
        L, inner = sx[ax], prod(sx[ax + 1:])
        cnt = prod(sx) // L
        fin = f'red[0] / {lit(L)}' if n.op_type == 'ReduceMean' else 'red[0]'
        main = (f'var<workgroup> red: array<f32, 64>;\n'
                f'@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n'
                f'  let o = wg.x;\n  let b = (o / {inner}u) * {L * inner}u + o % {inner}u;\n'
                f'  var s = 0.0; for (var r = li; r < {L}u; r += 64u) {{ s += {fx}(b + r * {inner}u); }}\n'
                f'  red[li] = s; workgroupBarrier();\n'
                f'  for (var w = 32u; w > 0u; w >>= 1u) {{ if (li < w) {{ red[li] += red[li + w]; }} workgroupBarrier(); }}\n'
                f'  if (li == 0u) {{ {store(k, R, fin, "o")} }}\n}}')
        return k, main, (cnt, 1, 1)

    for n in G.nodes:
        if G.is_view(n):
            continue
        for R in n.output:
            if not R or R in P.inline or R in P.absorbed or not G.uses[R]:
                continue
            if n.op_type in HEAVY:
                if n.op_type in ('Gemm', 'Einsum'):
                    k, main, wg = mm_kernel(R, R)
                elif n.op_type in ('Conv', 'ConvTranspose'):
                    k, main, wg = conv_kernel(R, R)
                else:
                    k, main, wg = red_kernel(R)
            elif R in P.fused:
                H = P.fused[R]
                if G.prod[H].op_type in ('Gemm', 'Einsum'):
                    k, main, wg = mm_kernel(R, H)
                else:
                    k, main, wg = conv_kernel(R, H)
            else:
                k, main, wg = ew_kernel(R)
            assert max(wg) <= 65535, (R, wg)
            kernels.append((k.source(main), k.slots, wg, R))
    return P, kernels


def layout(G, P, kernels):
    """arena: inputs, then outputs (both permanent), then a first-fit packing of every other materialized tensor by
    liveness; W: fp16-stored weights first (expanded on the GPU), then fp32 ones."""
    words = lambda t: (prod(G.shape[t]) + 3) // 4 if t == 'rgb' or G.dtype.get(t) == TensorProto.UINT8 else prod(G.shape[t])
    rnd = lambda n: (n + ALIGN - 1) // ALIGN * ALIGN
    off, top = {}, 0
    io = {}
    for t in G.inputs:
        off[t] = top
        io[t] = dict(offset=top, shape=list(G.shape[t]), dtype='int32' if G.dtype[t] == TensorProto.INT32 else 'float32')
        top += prod(G.shape[t])
    in_words = top
    top = rnd(top)
    out_start = top
    oio = {}
    for o in G.outputs:
        b = G.B(o)
        off[b] = top
        oio[o] = dict(offset=top, shape=list(G.shape[o]), dtype='uint8' if G.dtype[o] == TensorProto.UINT8 else 'float32')
        top += words(b)
    out_words = top - out_start
    top = rnd(top)
    first, last = {}, {}
    for d, (_, slots, _, R) in enumerate(kernels):
        first.setdefault(R, d)
        for kind, t in slots:
            if kind == 'A' and t != R:
                last[t] = d
    free, peak = [], top        # free: sorted [offset, size]
    by_last = defaultdict(list)
    for t, d in last.items():
        by_last[d].append(t)

    def alloc(n):
        nonlocal peak
        for j, (o, s) in enumerate(free):
            if s >= n:
                if s == n:
                    free.pop(j)
                else:
                    free[j] = [o + n, s - n]
                return o
        o = peak
        if free and free[-1][0] + free[-1][1] == peak:
            o = free.pop()[0]
        peak = o + n
        return o

    def release(o, n):
        free.append([o, n])
        free.sort()
        merged = []
        for a in free:
            if merged and merged[-1][0] + merged[-1][1] == a[0]:
                merged[-1][1] += a[1]
            else:
                merged.append(a)
        free[:] = merged

    for d, (_, _, _, R) in enumerate(kernels):
        if R not in off:
            off[R] = alloc(rnd(words(R)))
        for t in by_last[d]:
            if t not in G.inputs and t not in (G.B(o) for o in G.outputs):
                release(off[t], rnd(words(t)))
    for d, (_, slots, _, R) in enumerate(kernels):
        for kind, t in slots:
            assert kind in 'WHQS' or t in off, t
    # weights; --int8: each int8 tensor's scales ('S') sit with the fp32 weights, its bytes ('Q') in a third blob
    used = []
    for _, slots, _, _ in kernels:
        for kind, t in slots:
            key = t if kind in 'WH' else (kind, t)
            if kind != 'A' and key not in used:
                used.append(key)
    w16 = [t for t in used if not isinstance(t, tuple) and t in G.f16]
    w32 = [t for t in used if not isinstance(t, tuple) and t not in G.f16]
    w32 += [k for k in used if isinstance(k, tuple) and k[0] == 'S']
    w8 = [k for k in used if isinstance(k, tuple) and k[0] == 'Q']
    val32 = lambda k: G.q8[k[1]][3] if isinstance(k, tuple) else G.const[k]
    woff, top16 = {}, 0
    for t in w16:
        woff[t] = top16
        top16 += (prod(G.shape[t]) + 1) // 2 * 2
    top32 = 0 if OPT['f16'] else top16
    for t in w32:
        woff[t] = top32
        top32 += val32(t).size
    top8 = 0
    for k in w8:
        woff[k] = top8
        top8 += G.q8[k[1]][2].size
    blob16 = np.zeros(top16, np.float16)
    for t in w16:
        blob16[woff[t]:woff[t] + prod(G.shape[t])] = G.const[t].astype(np.float16).ravel()
    blob32 = np.concatenate([val32(t).astype(np.float32).ravel() for t in w32]) if w32 else np.zeros(0, np.float32)
    blob8 = np.zeros((top8 + 3) // 4 * 4, np.int8)
    for k in w8:
        blob8[woff[k]:woff[k] + G.q8[k[1]][2].size] = G.q8[k[1]][2].ravel()
    for t in w16:
        assert np.array_equal(G.const[t].astype(np.float16).astype(np.float32), G.const[t].astype(np.float32))
    params = []
    for _, slots, _, _ in kernels:
        params.append([off[t] if kind == 'A' else woff[t if kind in 'WH' else (kind, t)] for kind, t in slots])
    top32 = top32 if OPT['f16'] else top32 - top16
    return dict(off=off, woff=woff, inputs=io, outputs=oio, in_words=in_words, out_start=out_start, out_words=out_words,
                arena=peak, n16=top16, n32=top32, n8=blob8.size, params=params), blob16, blob32, blob8


def write_int8_onnx(G, src, out):
    """The ONNX graph with each int8 weight's fp16 initializer replaced by its dequantized values (fp32)."""
    m = onnx.load(src)
    casts = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == 'Cast'}
    inits = {i.name: i for i in m.graph.initializer}
    for t in G.q8:
        name = casts[t] if casts.get(t) in inits else t
        inits[name].CopyFrom(numpy_helper.from_array(G.const[t].astype(np.float32), name))
    onnx.save(m, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default=str(DOCS / 'play.onnx'))
    ap.add_argument('--out', default=str(DOCS / 'play.wgr'))
    ap.add_argument('--f32', action='store_true', help='expand the fp16 weights to fp32 on the GPU at load')
    ap.add_argument('--tile', default='{}', help='json overrides of TILE, for experiments')
    ap.add_argument('--int8', action='store_true', help='matmul and conv weights as int8 with fp32 scales (quantize())')
    ap.add_argument('--group', type=int, default=0, help='--int8: one scale per this many inputs (0: per output channel)')
    ap.add_argument('--int8-min', type=int, default=16384, help='--int8: smallest weight (elements) made int8')
    ap.add_argument('--int8-onnx', help='--int8: also write the ONNX graph with these int8 weights as fp32 (what this '
                                        'file computes; replay.py --onnx judges runs on it)')
    a = ap.parse_args()
    TILE.update({k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.tile).items()})
    OPT.update(f16=not a.f32, group=a.group)
    G = Graph(a.onnx)
    if a.int8:
        G.mark_int8(a.group, a.int8_min)
        if a.int8_onnx:
            write_int8_onnx(G, a.onnx, a.int8_onnx)
    P, kernels = build(G)
    L, b16, b32, b8 = layout(G, P, kernels)
    shaders, sidx, disp = [], {}, []
    for (src, slots, wg, R), prm in zip(kernels, L['params']):
        if src not in sidx:
            sidx[src] = len(shaders)
            shaders.append(src)
        disp.append([sidx[src], *wg])
    meta = dict(format='wgr1', f16=OPT['f16'], header=HEADER + (H16 if OPT['f16'] else '') + (Q8 if L['n8'] else ''),
                arena=L['arena'], n16=L['n16'], n32=L['n32'], **({'n8': L['n8']} if L['n8'] else {}), inputs=L['inputs'],
                outputs=L['outputs'], in_words=L['in_words'], out_start=L['out_start'], out_words=L['out_words'],
                shaders=shaders, dispatches=disp, params=L['params'], roots=[k[3] for k in kernels],
                root_off=[L['off'][k[3]] for k in kernels])
    js = json.dumps(meta, separators=(',', ':')).encode()
    Path(a.out + '.json').write_bytes(js)
    Path(a.out + '.bin').write_bytes(b16.tobytes() + b32.tobytes() + b8.tobytes())
    # one file for the browser: 'WGR1', u32 json bytes, json, zero padding to 4 bytes, fp16 weights, fp32 weights, and
    # with --int8 the int8 weights (n8 bytes, a multiple of 4)
    js += b' ' * (-len(js) % 4)
    Path(a.out).write_bytes(b'WGR1' + np.uint32(len(js)).tobytes() + js + b16.tobytes() + b32.tobytes() + b8.tobytes())
    ops = defaultdict(int)
    for k in kernels:
        ops[G.prod[k[3]].op_type if k[3] not in P.fused else G.prod[P.fused[k[3]]].op_type + '+epi'] += 1
    print(f'{len(G.nodes)} nodes -> {len(kernels)} dispatches, {len(shaders)} shaders; arena {L["arena"] * 4 / 1e6:.2f} MB; '
          f'weights {L["n16"]} fp16 + {L["n32"]} fp32 + {L["n8"]} int8 ({len(G.q8)} tensors) -> blob '
          f'{(b16.nbytes + b32.nbytes + b8.nbytes) / 1e6:.2f} MB, json '
          f'{Path(a.out + ".json").stat().st_size / 1e6:.2f} MB, {a.out}: {Path(a.out).stat().st_size / 1e6:.2f} MB')
    print(dict(sorted(ops.items(), key=lambda x: -x[1])))


if __name__ == '__main__':
    main()
