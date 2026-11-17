#!/usr/bin/env python3
"""GTRE v2 encoder: one BDL forest-district zip -> one compact .gtr file. numpy only.

    python3 gtre_encode.py BDL_07_34_ZAPOROWO_2026.zip out.gtr [--xy-cm 100] [--ids]
"""
import argparse
import lzma
import os
import re
import struct
import sys
import zipfile
import zlib
import numpy as np

MAGIC, VERSION = b"GTRE", 2
FLAG_IDS = 1
CODEC_RAW, CODEC_DEFLATE, CODEC_LZMA = 0, 1, 2
LZMA_DICT = 1 << 22
LZMA_VARIANTS = [(3, 0, 2), (0, 0, 0), (2, 0, 0)]
PALETTES = ("species", "part")


def enc_uvarints(a):
    a = np.asarray(a).astype(np.uint64)
    if a.size == 0:
        return b""
    nb = np.ones(a.shape, np.int64)
    t = a >> np.uint64(7)
    while True:
        m = t > 0
        if not m.any():
            break
        nb += m
        t = t >> np.uint64(7)
    off = np.cumsum(nb) - nb
    out = np.empty(int(nb.sum()), np.uint8)
    for i in range(int(nb.max())):
        m = nb > i
        byte = ((a[m] >> np.uint64(7 * i)) & np.uint64(127)).astype(np.uint8)
        byte |= ((nb[m] > i + 1).astype(np.uint8) << 7)
        out[off[m] + i] = byte
    return out.tobytes()


def enc_svarints(a):
    a = np.asarray(a, np.int64)
    return enc_uvarints(((a << 1) ^ (a >> 63)).astype(np.uint64))


def uv(v):
    return enc_uvarints(np.array([v], np.uint64))


def sv(v):
    return enc_svarints(np.array([v], np.int64))


def read_shp(buf):
    if struct.unpack(">i", buf[:4])[0] != 9994:
        raise ValueError("not a shapefile")
    stype = struct.unpack("<i", buf[32:36])[0]
    if stype not in (0, 5, 15, 25):
        raise ValueError(f"unsupported shape type {stype}")
    out, o, n = [], 100, len(buf)
    while o + 8 <= n:
        clen = struct.unpack(">i", buf[o + 4:o + 8])[0] * 2
        o += 8
        rec = buf[o:o + clen]
        o += clen
        t = struct.unpack_from("<i", rec, 0)[0]
        if t == 0:
            out.append(None)
            continue
        nparts, npts = struct.unpack_from("<ii", rec, 36)
        parts = np.frombuffer(rec, "<i4", nparts, 44).tolist() + [npts]
        pts = np.frombuffer(rec, "<f8", npts * 2, 44 + 4 * nparts).reshape(-1, 2)
        out.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return out


def read_dbf(buf, encoding="utf-8", want=("a_i_num", "area_type")):
    nrec, hlen, rlen = struct.unpack("<IHH", buf[4:12])
    fields, o = {}, 32
    pos = 1
    while buf[o] != 0x0D:
        name = buf[o:o + 11].split(b"\0")[0].decode("ascii", "replace").lower()
        flen = buf[o + 16]
        if name in want:
            fields[name] = (pos, flen)
        pos += flen
        o += 32
    if nrec == 0:
        return {k: [] for k in fields}
    rec = np.frombuffer(buf, np.uint8, nrec * rlen, hlen).reshape(nrec, rlen)
    deleted = rec[:, 0] == 0x2A
    out = {}
    for name, (p, l) in fields.items():
        col = np.ascontiguousarray(rec[:, p:p + l]).view(f"S{l}")[:, 0].tolist()
        out[name] = [None if d else c.decode(encoding, "replace").strip() for c, d in zip(col, deleted)]
    return out


def read_table(raw):
    for enc in ("utf-8-sig", "cp1250"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    lines = text.split("\n")
    head = [h.strip() for h in lines[0].rstrip("\r").split("\t")]
    rows = []
    for ln in lines[1:]:
        ln = ln.rstrip("\r")
        if ln:
            r = [c.strip() for c in ln.split("\t")]
            rows.append(r + [""] * (len(head) - len(r)))
    return head, rows


def table_dicts(raw, key, *vals):
    head, rows = read_table(raw)
    ix = {h: i for i, h in enumerate(head)}

    def d(s):
        return s[6:10] + s[3:5] + s[0:2] if len(s) == 10 else "99999999"

    best = {}
    for r in rows:
        k = (d(r[ix["date_to"]]), d(r[ix["date_from"]])) if "date_to" in ix else ("", "")
        c = r[ix[key]]
        if c and (c not in best or k >= best[c][0]):
            best[c] = (k, tuple(r[ix[v]] for v in vals))
    return {c: v for c, (_, v) in best.items()}


def _area2(p):
    p = p - p[0]
    return float((p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]).sum())


def prep_ring(g):
    if len(g) > 1 and (g[0] == g[-1]).all():
        g = g[:-1]
    if len(g) < 3:
        return None
    keep = np.ones(len(g), bool)
    keep[1:] = (g[1:] != g[:-1]).any(1)
    g = g[keep]
    while len(g) > 1 and (g[0] == g[-1]).all():
        g = g[:-1]
    if len(g) < 3:
        return None
    for _ in range(3):
        d1 = g - np.roll(g, 1, 0)
        d2 = np.roll(g, -1, 0) - g
        col = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] == 0) & ((d1 * d2).sum(1) > 0)
        col[0] = False
        if not col.any():
            break
        g = g[~col]
        if len(g) < 3:
            return None
    return g


def prep_stand(rings_f, scale):
    areas = [_area2(r) for r in rings_f if len(r) >= 4]
    flip = sum(areas) > 0
    out = []
    for r in rings_f:
        if len(r) < 4:
            continue
        a = _area2(r)
        hole = (a < 0) if flip else (a > 0)
        g = prep_ring(np.rint(r * scale).astype(np.int64))
        if g is None:
            continue
        q = _area2(g)
        if q == 0:
            continue
        if (q > 0) == hole:
            g = np.concatenate([g[:1], g[:0:-1]])
        out.append((hole, g))
    return out


def morton(x, y):
    def spread(v):
        v = v.astype(np.uint64) & np.uint64(0xFFFFFFFF)
        for s, m in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF),
                     (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333), (1, 0x5555555555555555)):
            v = (v | (v << np.uint64(s))) & np.uint64(m)
        return v
    return spread(x) | (spread(y) << np.uint64(1))


# ---------- height models ----------
_NASLUND = {
    "SO": (1.85, 0.145), "SOB": (1.90, 0.150), "SOCZ": (1.80, 0.140),
    "SOSM": (1.85, 0.145), "SOWM": (1.75, 0.135),
    "SW": (1.70, 0.130), "ŚW": (1.70, 0.130), "JD": (1.90, 0.140),
    "DG": (1.60, 0.125), "MD": (1.55, 0.125),
    "DB": (2.00, 0.150), "DBS": (2.00, 0.150), "DBC": (2.05, 0.155), "DBB": (2.00, 0.150),
    "BK": (1.75, 0.135), "GB": (1.80, 0.145),
    "BRZ": (1.60, 0.140), "BRZB": (1.60, 0.140), "BRZOM": (1.65, 0.145),
    "OL": (1.65, 0.145), "OLSZ": (1.70, 0.150),
    "TP": (1.50, 0.130), "OS": (1.55, 0.135), "WB": (1.45, 0.140),
    "KL": (1.70, 0.140), "JKL": (1.65, 0.145), "JW": (1.70, 0.140),
    "JS": (1.75, 0.145), "LP": (1.80, 0.150),
    "AK": (1.50, 0.160), "CZM": (1.55, 0.155), "CZR": (1.60, 0.150),
    "GR": (1.70, 0.155), "JB": (1.65, 0.150), "JRZ": (1.55, 0.145),
    "KSZ": (1.60, 0.150), "ORZ": (1.70, 0.145), "WZ": (1.75, 0.145),
}

_CR_BASE = {
    "SO": (36.0, 0.022, 1.40), "SOB": (28.0, 0.028, 1.30), "SOCZ": (34.0, 0.021, 1.40),
    "SOSM": (32.0, 0.022, 1.35), "SOWM": (35.0, 0.023, 1.35),
    "SW": (38.0, 0.024, 1.35), "ŚW": (38.0, 0.024, 1.35), "JD": (34.0, 0.018, 1.50),
    "DG": (38.0, 0.025, 1.30), "MD": (40.0, 0.026, 1.30),
    "DB": (32.0, 0.018, 1.50), "DBS": (32.0, 0.018, 1.50), "DBC": (30.0, 0.019, 1.45), "DBB": (31.0, 0.017, 1.50),
    "BK": (34.0, 0.020, 1.45), "GB": (27.0, 0.022, 1.40),
    "BRZ": (30.0, 0.030, 1.25), "BRZB": (30.0, 0.030, 1.25), "BRZOM": (28.0, 0.028, 1.30),
    "OL": (28.0, 0.028, 1.30), "OLSZ": (24.0, 0.026, 1.30),
    "TP": (32.0, 0.035, 1.20), "OS": (29.0, 0.032, 1.20), "WB": (22.0, 0.040, 1.15),
    "KL": (28.0, 0.022, 1.35), "JKL": (25.0, 0.028, 1.25), "JW": (30.0, 0.022, 1.35),
    "JS": (30.0, 0.020, 1.40), "LP": (28.0, 0.020, 1.40),
    "AK": (24.0, 0.035, 1.20), "CZM": (18.0, 0.040, 1.20), "CZR": (20.0, 0.035, 1.25),
    "GR": (18.0, 0.030, 1.30), "JB": (16.0, 0.032, 1.25), "JRZ": (15.0, 0.035, 1.25),
    "KSZ": (22.0, 0.025, 1.35), "ORZ": (25.0, 0.022, 1.40), "WZ": (30.0, 0.020, 1.40),
}

_SITE_SCALE = {
    "I": 1.00, "1": 1.00, "IA": 1.05, "IB": 0.98,
    "II": 0.90, "2": 0.90, "IIA": 0.93, "IIB": 0.87,
    "III": 0.78, "3": 0.78, "IIIA": 0.82, "IIIB": 0.74,
    "IV": 0.65, "4": 0.65, "IVA": 0.68, "IVB": 0.60,
    "V": 0.52, "5": 0.52, "VA": 0.55, "VB": 0.48,
}


def model_naslunda(species_cd, bhd):
    if bhd is None or bhd <= 0:
        return None
    a, b = _NASLUND.get((species_cd or "").upper(), (1.80, 0.140))
    d = float(bhd)
    h = 1.3 + (d / (a + b * d)) ** 2
    return max(1.5, min(45.0, h))


def model_chapmana_richardsa(species_cd, age, site_class_cd=None):
    if age is None or age <= 0:
        return None
    age = float(age)
    Hmax, k, p = _CR_BASE.get((species_cd or "").upper(), (30.0, 0.022, 1.40))
    Hmax *= _SITE_SCALE.get((site_class_cd or "").strip().upper(), 0.85)
    h = Hmax * (1.0 - np.exp(-k * age)) ** p
    return max(1.0, min(Hmax * 1.05, h))


def wyznacz_wysokosc_bdl(species_cd, height, bhd, age, site_class_cd):
    if height is not None and height > 0:
        return float(height), False
    if bhd is not None and bhd > 0:
        h = model_naslunda(species_cd, bhd)
        if h is not None:
            return round(h, 1), True
    if age is not None and age > 0:
        h = model_chapmana_richardsa(species_cd, age, site_class_cd)
        if h is not None:
            return round(h, 1), True
    return None, False


def _find(zf):
    return {os.path.basename(n).lower(): n for n in zf.namelist() if not n.endswith("/")}


TREE_AREA_TYPES = {
    "D-STAN",   # tree stands
    "PLANT CH", # tree plantations
    "PLANT SZ", # fast-growing tree plantations
}


def load_zip(path, keep_empty=False):
    m = re.search(r"BDL_(\d+)_(\d+)_(.+)_(\d{4})", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read rdlp/nadl/year from file name: {path}")
    meta = dict(rdlp=int(m[1]), nadl=int(m[2]), name=m[3], year=int(m[4]))
    zf = zipfile.ZipFile(path)
    f = _find(zf)
    for need in ("g_subarea.shp", "g_subarea.dbf", "f_storey_species.txt"):
        if need not in f:
            raise ValueError(f"{need} missing in {path}")
    prj = zf.read(f["g_subarea.prj"]).decode("ascii", "replace") if "g_subarea.prj" in f else ""
    if prj and not re.search(r"CS92|1992|2180", prj):
        raise ValueError(f"unexpected CRS: {prj[:80]}")
    cpg = zf.read(f["g_subarea.cpg"]).decode("ascii", "replace").lower() if "g_subarea.cpg" in f else "utf-8"
    enc = "cp1250" if "1250" in cpg else "iso-8859-2" if "8859-2" in cpg else "utf-8"

    geoms = read_shp(zf.read(f["g_subarea.shp"]))
    dbf = read_dbf(zf.read(f["g_subarea.dbf"]), enc)
    if len(dbf["a_i_num"]) != len(geoms):
        raise ValueError("shp/dbf mismatch")

    head, rows = read_table(zf.read(f["f_storey_species.txt"]))
    ix = {h: i for i, h in enumerate(head)}
    sto_order = {}
    if "f_storey_dic.txt" in f:
        sto_order = {c: int(v[0] or 99) for c, v in table_dicts(zf.read(f["f_storey_dic.txt"]), "storey_cd", "storey_nr").items()}
    by_id = {}
    for r in rows:
        by_id.setdefault(r[ix["arodes_int_num"]], []).append(r)

    def num(s, scale=1.0):
        return None if s == "" else float(s) * scale

    stands, n_non_forest = [], 0
    n_from_bdl = n_from_bhd = n_from_age = n_missing = 0

    for aid, typ, rg in zip(dbf["a_i_num"], dbf["area_type"], geoms):
        if aid is None or rg is None:
            continue
        typ = (typ or "").strip().upper()

        if typ not in TREE_AREA_TYPES:
            n_non_forest += 1
            continue

        rr = by_id.get(aid)
        if not rr and not keep_empty:
            continue
        rr = sorted(rr or [], key=lambda r: (sto_order.get(r[ix["storey_cd"]], 99),
                                             int(r[ix["sp_rank_order_act"]] or 999)))
        trees = []
        for r in rr:
            sp = r[ix["species_cd"]]
            age = num(r[ix["species_age"]])
            part = r[ix["part_cd_act"]]
            h_bdl = num(r[ix["height"]])
            bhd = num(r[ix["bhd"]])
            site = r[ix["site_class_cd"]]
            h, est = wyznacz_wysokosc_bdl(sp, h_bdl, bhd, age, site)
            if not est and h is not None:
                n_from_bdl += 1
            elif est and bhd is not None and bhd > 0 and (h_bdl is None or h_bdl <= 0):
                n_from_bhd += 1
            elif est:
                n_from_age += 1
            else:
                n_missing += 1
            trees.append((sp, age, part, h, est))
        stands.append(dict(id=int(aid), rings_f=rg, rows=trees))

    dicts = {}
    if "f_tree_species_dic.txt" in f:
        dicts["species"] = {c: list(v) for c, v in table_dicts(
            zf.read(f["f_tree_species_dic.txt"]), "species_cd", "species_name", "latin_name").items()}
    if "f_part_dic.txt" in f:
        dicts["part"] = {c: v[0] for c, v in table_dicts(
            zf.read(f["f_part_dic.txt"]), "part_cd", "part_name").items()}

    meta["n_non_forest_dropped"] = n_non_forest
    meta["height_stats"] = dict(bdl=n_from_bdl, from_bhd=n_from_bhd, from_age=n_from_age, missing=n_missing)
    return meta, stands, dicts


def pack_stream(raw, fast=False):
    best, codec = raw, CODEC_RAW
    if len(raw) >= 16:
        co = zlib.compressobj(9, zlib.DEFLATED, -15)
        z = co.compress(raw) + co.flush()
        if len(z) < len(best):
            best, codec = z, CODEC_DEFLATE
        if not fast:
            for lc, lp, pb in LZMA_VARIANTS:
                filt = [dict(id=lzma.FILTER_LZMA2, preset=9 | lzma.PRESET_EXTREME,
                             dict_size=LZMA_DICT, lc=lc, lp=lp, pb=pb)]
                z = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filt)
                if len(z) < len(best):
                    best, codec = z, CODEC_LZMA
    return uv((len(best) << 2) | codec) + best


def _pack_bitset(bits):
    bits = list(bits)
    if not bits:
        return b""
    n = (len(bits) + 7) // 8
    out = bytearray(n)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def build(meta, stands, xy_cm=100, tile_m=4096, block_stands=20000, ids=False, fast=False):
    if not 1 <= xy_cm <= 255:
        raise ValueError("xy_cm must be 1..255")
    scale = 100.0 / xy_cm
    tile_steps = max(1, int(round(tile_m * 100 / xy_cm)))
    st = dict(shp_stands=len(stands), degenerate=0,
              non_forest_dropped=meta.get("n_non_forest_dropped", 0),
              height_stats=meta.get("height_stats", {}))

    S = []
    for s in stands:
        rings = prep_stand(s["rings_f"], scale)
        if not rings:
            st["degenerate"] += 1
            continue
        allp = np.concatenate([g for _, g in rings])
        S.append(dict(s, rings=rings, bbox=(*allp.min(0), *allp.max(0))))
    st["stands"] = len(S)
    st["rows"] = sum(len(s["rows"]) for s in S)
    st["vertices"] = sum(len(g) for s in S for _, g in s["rings"])
    st["estimated_heights"] = sum(1 for s in S for r in s["rows"] if r[4])

    cnt = {k: {} for k in PALETTES}
    for s in S:
        for r in s["rows"]:
            for k, v in zip(("species", "part"), (r[0], r[2])):
                cnt[k][v] = cnt[k].get(v, 0) + 1
    pal, pix = {}, {}
    for k, c in cnt.items():
        pal[k] = [v for v, _ in sorted(((v, n) for v, n in c.items() if v), key=lambda t: (-t[1], t[0]))]
        pix[k] = {v: i + 1 for i, v in enumerate(pal[k])}
        pix[k][""] = 0

    if S:
        anc = np.array([s["rings"][0][1][0] for s in S], np.int64)
        tx = np.floor_divide(anc[:, 0], tile_steps)
        ty = np.floor_divide(anc[:, 1], tile_steps)
        off = anc - np.stack([tx, ty], 1) * tile_steps
        order = np.lexsort((morton(off[:, 0], off[:, 1]), morton(tx - tx.min(), ty - ty.min())))
    else:
        tx = ty = np.zeros(0, np.int64)
        order = np.zeros(0, np.int64)

    tiles = []
    for i in order:
        if tiles and tiles[-1][0] == tx[i] and tiles[-1][1] == ty[i]:
            tiles[-1][2].append(int(i))
        else:
            tiles.append([int(tx[i]), int(ty[i]), [int(i)]])

    blocks, cur, n = [], [], 0
    for t in tiles:
        cur.append(t)
        n += len(t[2])
        if n >= block_stands:
            blocks.append(cur)
            cur, n = [], 0
    if cur:
        blocks.append(cur)

    flags = FLAG_IDS if ids else 0
    head = bytearray(MAGIC + bytes([VERSION, flags, xy_cm, meta["rdlp"], meta["nadl"]]))
    head += uv(meta["year"]) + uv(tile_steps)
    for k in PALETTES:
        head += uv(len(pal[k]))
        for v in pal[k]:
            b = v.encode("utf-8")
            head += uv(len(b)) + b

    head += uv(len(tiles))
    ptx = pty = 0
    bb = [1 << 60, 1 << 60, -(1 << 60), -(1 << 60)]
    for tx_, ty_, idx in tiles:
        x0 = min(S[i]["bbox"][0] for i in idx) - tx_ * tile_steps
        y0 = min(S[i]["bbox"][1] for i in idx) - ty_ * tile_steps
        x1 = max(S[i]["bbox"][2] for i in idx) - tx_ * tile_steps
        y1 = max(S[i]["bbox"][3] for i in idx) - ty_ * tile_steps
        head += sv(tx_ - ptx) + sv(ty_ - pty) + sv(x0) + sv(y0) + uv(x1 - x0) + uv(y1 - y0) + uv(len(idx))
        ptx, pty = tx_, ty_
        bb = [min(bb[0], x0 + tx_ * tile_steps), min(bb[1], y0 + ty_ * tile_steps),
              max(bb[2], x1 + tx_ * tile_steps), max(bb[3], y1 + ty_ * tile_steps)]
    st["bbox_m"] = [round(v * xy_cm / 100.0, 1) for v in bb] if tiles else None

    body, btab = [], bytearray()
    btab += uv(len(blocks))
    for blk in blocks:
        prev_id = 0
        A = {k: [] for k in ("ax", "ay", "rings", "vx", "vy", "nrow", "ids",
                             "species", "age", "part", "h", "hsrc")}
        for tx_, ty_, idx in blk:
            ox, oy = tx_ * tile_steps, ty_ * tile_steps
            px = py = 0
            for i in idx:
                s = S[i]
                a = s["rings"][0][1][0]
                ax_, ay_ = int(a[0]) - ox, int(a[1]) - oy
                A["ax"].append(ax_ - px)
                A["ay"].append(ay_ - py)
                px, py = ax_, ay_
                prev = a
                for k, (hole, g) in enumerate(s["rings"]):
                    A["rings"].append((len(g) << 2) | (int(hole) << 1) | int(k == len(s["rings"]) - 1))
                    if k:
                        A["vx"].append(np.array([g[0, 0] - prev[0]]))
                        A["vy"].append(np.array([g[0, 1] - prev[1]]))
                        prev = g[0]
                    d = np.diff(g, axis=0)
                    A["vx"].append(d[:, 0])
                    A["vy"].append(d[:, 1])
                A["nrow"].append(len(s["rows"]))
                if ids:
                    A["ids"].append(s["id"] - prev_id)
                    prev_id = s["id"]
                for r in s["rows"]:
                    A["species"].append(pix["species"][r[0]])
                    A["age"].append(0 if r[1] is None else int(round(r[1])) + 1)
                    A["part"].append(pix["part"][r[2]])
                    A["h"].append(0 if r[3] is None else int(round(r[3])) + 1)
                    A["hsrc"].append(1 if r[4] else 0)
        cat = lambda l: np.concatenate(l) if l else np.zeros(0, np.int64)
        streams = [enc_svarints(A["ax"]), enc_svarints(A["ay"]), enc_uvarints(A["rings"]),
                   enc_svarints(cat(A["vx"])), enc_svarints(cat(A["vy"])), enc_uvarints(A["nrow"])]
        if ids:
            streams.append(enc_svarints(A["ids"]))
        streams += [enc_uvarints(A[k]) for k in ("species", "age", "part", "h")]
        streams.append(_pack_bitset(A["hsrc"]))
        blob = b"".join(pack_stream(s, fast) for s in streams)
        body.append(blob)
        btab += uv(len(blk)) + uv(len(blob))
    return bytes(head) + bytes(btab) + b"".join(body), st


def encode_zip(path, out=None, keep_empty=False, **kw):
    meta, stands, dicts = load_zip(path, keep_empty)
    data, st = build(meta, stands, **kw)
    if out:
        with open(out, "wb") as fh:
            fh.write(data)
    st.update(meta=meta, bytes=len(data))
    return data, st, dicts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("out")
    ap.add_argument("--xy-cm", type=int, default=100)
    ap.add_argument("--tile-m", type=int, default=4096)
    ap.add_argument("--block-stands", type=int, default=20000)
    ap.add_argument("--ids", action="store_true")
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    _, st, _ = encode_zip(a.zip, a.out, a.keep_empty, xy_cm=a.xy_cm, tile_m=a.tile_m,
                          block_stands=a.block_stands, ids=a.ids, fast=a.fast)
    hs = st.get("height_stats", {})
    print(f"{st['stands']} stands ({st['degenerate']} degenerate, {st.get('non_forest_dropped', 0)} non-forest), "
          f"{st['rows']} rows ({st.get('estimated_heights', 0)} est) -> {st['bytes']} bytes")
    if hs:
        print(f"  heights: BDL={hs.get('bdl',0)} BHD={hs.get('from_bhd',0)} age={hs.get('from_age',0)} miss={hs.get('missing',0)}")