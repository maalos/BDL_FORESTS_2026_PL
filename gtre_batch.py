#!/usr/bin/env python3
"""Encode every BDL_*.zip under a directory into .gtr files.

    python3 gtre_batch.py ~/Downloads/BDL_2026 out/ -j 8 --ids
"""
import argparse, json, os, re, sys, time, traceback
from multiprocessing import Pool
import numpy as np
import gtre_encode as enc
import pl1992

PAT = re.compile(r"^BDL_(\d+)_(\d+)_(.+)_(\d{4})\.zip$", re.I)


def find_zips(root):
    found = []
    for d, _, files in os.walk(root):
        found += [os.path.join(d, f) for f in files if PAT.match(f)]
    return sorted(found, key=lambda p: tuple(
        int(x) if x.isdigit() else x for x in PAT.match(os.path.basename(p)).groups()))


def work(job):
    zp, out_root, opts, keep_empty, force = job
    stem = os.path.basename(zp)[:-4]
    rdlp = PAT.match(os.path.basename(zp))[1]
    out = os.path.join(out_root, str(int(rdlp)), stem + ".gtr")
    meta = os.path.join(out_root, "_meta", stem + ".json")
    try:
        if not force and os.path.exists(out) and os.path.exists(meta):
            return stem, "skipped", None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.makedirs(os.path.dirname(meta), exist_ok=True)
        t = time.time()
        _, st, dicts = enc.encode_zip(zp, out + ".part", keep_empty, **opts)
        os.replace(out + ".part", out)
        st["seconds"] = round(time.time() - t, 1)
        st["file"] = os.path.relpath(out, out_root).replace(os.sep, "/")
        st["zip_bytes"] = os.path.getsize(zp)
        with open(meta + ".part", "w", encoding="utf-8") as fh:
            json.dump(dict(stats=st, dicts=dicts), fh, ensure_ascii=False)
        os.replace(meta + ".part", meta)
        return stem, "ok", st
    except Exception:
        return stem, "error", traceback.format_exc()


def lonlat_bbox(b):
    x = np.array([b[0], b[2], b[0], b[2], (b[0]+b[2])/2, (b[0]+b[2])/2, b[0], b[2]])
    y = np.array([b[1], b[1], b[3], b[3], b[1], b[3], (b[1]+b[3])/2, (b[1]+b[3])/2])
    lon, lat = pl1992.to_lonlat(x, y)
    return [round(float(lon.min()), 5), round(float(lat.min()), 5),
            round(float(lon.max()), 5), round(float(lat.max()), 5)]


def build_index(out_root):
    index, dicts, conflicts = [], {}, 0
    mdir = os.path.join(out_root, "_meta")
    for fn in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
        if not fn.endswith(".json"):
            continue
        m = json.load(open(os.path.join(mdir, fn), encoding="utf-8"))
        st = m["stats"]
        ent = dict(file=st["file"], rdlp=st["meta"]["rdlp"], nadl=st["meta"]["nadl"],
                   name=st["meta"]["name"], year=st["meta"]["year"], bytes=st["bytes"],
                   stands=st["stands"], rows=st["rows"], vertices=st["vertices"],
                   degenerate=st["degenerate"], bbox_m=st["bbox_m"],
                   bbox_lonlat=lonlat_bbox(st["bbox_m"]) if st["bbox_m"] else None)
        index.append(ent)
        for cat, d in m["dicts"].items():
            tgt = dicts.setdefault(cat, {})
            for code, val in d.items():
                if code in tgt and tgt[code] != val:
                    conflicts += 1
                else:
                    tgt.setdefault(code, val)
    index.sort(key=lambda e: (e["rdlp"], e["nadl"]))
    json.dump(index, open(os.path.join(out_root, "index.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    json.dump(dicts, open(os.path.join(out_root, "dictionaries.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)
    return index, dicts, conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--xy-cm", type=int, default=100)
    ap.add_argument("--tile-m", type=int, default=4096)
    ap.add_argument("--block-stands", type=int, default=20000)
    ap.add_argument("--ids", action="store_true")
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    zips = find_zips(a.src)
    if not zips:
        sys.exit(f"no BDL_*_*_*_<year>.zip found under {a.src}")
    opts = dict(xy_cm=a.xy_cm, tile_m=a.tile_m, block_stands=a.block_stands,
                ids=a.ids, fast=a.fast)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(z, a.out, opts, a.keep_empty, a.force) for z in zips]
    print(f"{len(zips)} zips, {a.jobs} workers", flush=True)

    errors, done, t0 = [], 0, time.time()
    with Pool(a.jobs) as pool:
        for stem, status, info in pool.imap_unordered(work, jobs):
            done += 1
            if status == "error":
                errors.append((stem, info))
                print(f"[{done}/{len(zips)}] {stem}  ERROR", flush=True)
            elif status == "skipped":
                print(f"[{done}/{len(zips)}] {stem}  skipped", flush=True)
            else:
                print(f"[{done}/{len(zips)}] {stem}  {info['stands']} stands, "
                      f"{info['rows']} rows -> {info['bytes']/1024:.0f} KiB ({info['seconds']} s)", flush=True)

    index, dicts, conflicts = build_index(a.out)
    tot_b = sum(e["bytes"] for e in index)
    print(f"\n{len(index)} files, {sum(e['stands'] for e in index)} stands, "
          f"{sum(e['rows'] for e in index)} rows, {tot_b/1e6:.1f} MB, {time.time()-t0:.0f} s")
    if conflicts:
        print(f"warning: {conflicts} dict conflicts (first kept)")
    if errors:
        with open(os.path.join(a.out, "errors.log"), "w", encoding="utf-8") as fh:
            for stem, tb in errors:
                fh.write(f"== {stem}\n{tb}\n")
        print(f"{len(errors)} failed -> errors.log")
        sys.exit(1)


if __name__ == "__main__":
    main()