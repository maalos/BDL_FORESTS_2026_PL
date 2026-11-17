#!/usr/bin/env python3
"""Przetwarza rekursywnie wszystkie zipy BDL w cwd -> .gtr.

    cd ~/Downloads/polishtrees
    python3 process_all.py
    python3 process_all.py --force --fast
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gtre_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.getcwd())
    ap.add_argument("--out", default="gtre_out")
    ap.add_argument("--no-ids", action="store_true")
    a, rest = ap.parse_known_args()

    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out)
    zips = gtre_batch.find_zips(src)
    zips = [z for z in zips if not os.path.abspath(z).startswith(out + os.sep)]

    print(f"katalog: {src}\nwyjście: {out}\nzipów BDL: {len(zips)}", flush=True)
    if not zips:
        sys.exit("nie znaleziono BDL_*.zip")

    argv = [sys.argv[0], src, out] + ([] if a.no_ids else ["--ids"]) + rest
    sys.argv = argv
    gtre_batch.main()


if __name__ == "__main__":
    main()