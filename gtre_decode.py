#!/usr/bin/env python3
"""Reference decoder for GTRE v2. numpy only.

    python3 gtre_decode.py file.gtr
    python3 gtre_decode.py file.gtr --at 362294 362199
"""
import lzma, sys, zlib
from collections import Counter
import numpy as np

LZMA_FILTERS = [dict(id=lzma.FILTER_LZMA2, dict_size=1 << 22)]
PALETTES = ("species", "part")
FLAG_IDS = 1


def _uv(b, o):
    r = s = 0
    while True:
        c = b[o]; o += 1
        r |= (c & 127) << s
        if c < 128:
            return r, o
        s += 7


def _sv(b, o):
    v, o = _uv(b, o)
    return (v >> 1) ^ -(v & 1), o


def _varints(buf):
    a = np.frombuffer(buf, np.uint8)
    if not len(a):
        return np.zeros(0, np.int64)
    ends = np.flatnonzero(a < 128)
    starts = np.concatenate(([0], ends[:-1] + 1))
    ln = ends - starts + 1
    out = np.zeros(len(ends), np.int64)
    for i in range(int(ln.max())):
        m = ln > i
        out[m] |= (a[starts[m] + i].astype(np.int64) & 127) << (7 * i)
    return out


def _zz(v):
    return (v >> 1) ^ -(v & 1)


def _unpack(b, o):
    v, o = _uv(b, o)
    n, codec = v >> 2, v & 3
    p = bytes(b[o:o + n])
    if codec == 1:
        p = zlib.decompress(p, -15)
    elif codec == 2:
        p = lzma.decompress(p, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    elif codec != 0:
        raise ValueError(f"unknown codec {codec}")
    return p, o + n


def _unpack_bitset(raw, nbits):
    out = []
    for i in range(nbits):
        byte = raw[i // 8] if i // 8 < len(raw) else 0
        out.append((byte >> (i % 8)) & 1)
    return out


def read_header(b):
    if bytes(b[:4]) != b"GTRE" or b[4] != 2:
        raise ValueError("not a GTRE v2 file")
    flags, xy_cm, rdlp, nadl = b[5], b[6], b[7], b[8]
    year, o = _uv(b, 9)
    tile_steps, o = _uv(b, o)
    pal = {}
    for k in PALETTES:
        n, o = _uv(b, o)
        items = []
        for _ in range(n):
            ln, o = _uv(b, o)
            items.append(bytes(b[o:o + ln]).decode("utf-8"))
            o += ln
        pal[k] = items
    nt, o = _uv(b, o)
    tiles, px, py = [], 0, 0
    for _ in range(nt):
        dx, o = _sv(b, o); dy, o = _sv(b, o)
        x0, o = _sv(b, o); y0, o = _sv(b, o)
        w, o = _uv(b, o); h, o = _uv(b, o); c, o = _uv(b, o)
        px += dx; py += dy
        tiles.append((px, py, x0, y0, w, h, c))
    nb, o = _uv(b, o)
    blocks = []
    for _ in range(nb):
        t, o = _uv(b, o); ln, o = _uv(b, o)
        blocks.append((t, ln))
    names = ["ax", "ay", "rings", "vx", "vy", "nrow"]
    if flags & FLAG_IDS:
        names.append("ids")
    names += ["species", "age", "part", "h", "hsrc"]
    return dict(flags=flags, xy_cm=xy_cm, rdlp=rdlp, nadl=nadl, year=year,
                tile_steps=tile_steps, palettes=pal, tiles=tiles, blocks=blocks,
                data=o, streams=names)


def tile_bbox_m(h, ti):
    tx, ty, x0, y0, w, hh, _ = h["tiles"][ti]
    s, ts = h["xy_cm"] / 100.0, h["tile_steps"]
    return ((tx * ts + x0) * s, (ty * ts + y0) * s,
            (tx * ts + x0 + w) * s, (ty * ts + y0 + hh) * s)


def _decode_block(b, o, ln, tcount, t0, h):
    pal, s = h["palettes"], h["xy_cm"] / 100.0
    raw, p = {}, o
    for nm in h["streams"]:
        raw[nm], p = _unpack(b, p)
    if p != o + ln:
        raise ValueError("block length mismatch")
    ax = _zz(_varints(raw["ax"])).tolist()
    ay = _zz(_varints(raw["ay"])).tolist()
    rg = _varints(raw["rings"]).tolist()
    nrow = _varints(raw["nrow"]).tolist()
    vx, vy = _zz(_varints(raw["vx"])), _zz(_varints(raw["vy"]))
    ids = _zz(_varints(raw["ids"])).tolist() if "ids" in raw else None
    col = {k: _varints(raw[k]).tolist() for k in ("species", "age", "part", "h")}
    hsrc = _unpack_bitset(raw["hsrc"], sum(nrow))

    def lab(k, i):
        return pal[k][i - 1] if i else ""

    i = rp = vp = rowp = 0
    prev_id = 0
    for ti in range(t0, t0 + tcount):
        tx, ty, *_, cnt = h["tiles"][ti]
        ox, oy = tx * h["tile_steps"], ty * h["tile_steps"]
        gx = gy = 0
        for _ in range(cnt):
            gx += ax[i]; gy += ay[i]
            sx, sy, k, rings = gx, gy, 0, []
            while True:
                v = rg[rp]; rp += 1
                n, hole, last = v >> 2, (v >> 1) & 1, v & 1
                if k:
                    sx += int(vx[vp]); sy += int(vy[vp]); vp += 1
                pts = np.empty((n, 2), np.int64)
                pts[0] = (sx, sy)
                pts[1:, 0] = sx + np.cumsum(vx[vp:vp + n - 1])
                pts[1:, 1] = sy + np.cumsum(vy[vp:vp + n - 1])
                vp += n - 1
                rings.append((bool(hole), (pts + (ox, oy)) * s))
                k += 1
                if last:
                    break
            trees = []
            for _r in range(nrow[i]):
                a, hh = col["age"][rowp], col["h"][rowp]
                trees.append(dict(
                    species=lab("species", col["species"][rowp]),
                    age=a - 1 if a else None,
                    part=lab("part", col["part"][rowp]),
                    height=hh - 1 if hh else None,
                    height_estimated=bool(hsrc[rowp]),
                ))
                rowp += 1
            st = dict(tile=ti, rings=rings, trees=trees)
            if ids is not None:
                prev_id += ids[i]
                st["id"] = prev_id
            yield st
            i += 1


def read_stands(path_or_bytes, bbox=None):
    b = open(path_or_bytes, "rb").read() if isinstance(path_or_bytes, str) else path_or_bytes
    h = read_header(b)
    o, t0 = h["data"], 0
    for tcount, ln in h["blocks"]:
        if bbox is not None:
            tb = [tile_bbox_m(h, t) for t in range(t0, t0 + tcount)]
            if not any(t[0] <= bbox[2] and t[2] >= bbox[0] and t[1] <= bbox[3] and t[3] >= bbox[1] for t in tb):
                o += ln; t0 += tcount; continue
        for st in _decode_block(b, o, ln, tcount, t0, h):
            if bbox is not None:
                pts = np.concatenate([p for _, p in st["rings"]])
                lo, hi = pts.min(0), pts.max(0)
                if lo[0] > bbox[2] or hi[0] < bbox[0] or lo[1] > bbox[3] or hi[1] < bbox[1]:
                    continue
            yield st
        o += ln; t0 += tcount


def _inside(ring, x, y):
    x1, y1 = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    cross = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(cross & (x < xi)) & 1)


def stand_at(path_or_bytes, x, y):
    for st in read_stands(path_or_bytes, bbox=(x, y, x, y)):
        if sum(_inside(r, x, y) for _, r in st["rings"]) & 1:
            return st
    return None


def summarize(path):
    b = open(path, "rb").read()
    h = read_header(b)
    n = rows = est = 0
    sp = Counter()
    for st in read_stands(b):
        n += 1
        rows += len(st["trees"])
        for t in st["trees"]:
            sp[t["species"]] += 1
            if t.get("height_estimated"):
                est += 1
    print(f"{path}: {len(b)} B, rdlp {h['rdlp']} nadl {h['nadl']} year {h['year']}, "
          f"{len(h['tiles'])} tiles, {len(h['blocks'])} blocks")
    print(f"{n} stands, {rows} rows ({est} est); species: {sp.most_common(8)}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[2] == "--at":
        s = stand_at(sys.argv[1], float(sys.argv[3]), float(sys.argv[4]))
        print(s if s is None else {k: v for k, v in s.items() if k != "rings"})
    else:
        summarize(sys.argv[1])