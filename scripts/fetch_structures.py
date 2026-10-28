import os, csv, argparse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "structures"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    seen = set()
    with open(os.path.join(ROOT, "data", "pdb_pairs.csv")) as f:
        for row in csv.DictReader(f):
            for pdb in (row["pdb_1"], row["pdb_2"]):
                if pdb in seen: continue
                seen.add(pdb); dst = os.path.join(a.out, pdb + ".cif")
                if os.path.exists(dst): continue
                url = "https://files.rcsb.org/download/%s.cif" % pdb
                print("fetch", pdb); urllib.request.urlretrieve(url, dst)
    print("structures=%d out=%s" % (len(seen), a.out))

if __name__ == "__main__":
    main()
