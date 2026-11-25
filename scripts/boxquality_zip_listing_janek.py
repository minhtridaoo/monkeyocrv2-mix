"""Read ONLY the central directory of the official TextVQA train_val_images.zip via HTTP Range requests (no full download)
and check that every selected TRAIN image (docs/handoff/artifacts/janek/boxquality_p0_20261124/selection_janek.csv) is a member with the expected name.
Output: docs/handoff/artifacts/janek/boxquality_p0_20261124/zip_listing_janek.json"""
import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
BYTES = 7072297970


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.fetched = url, size, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}', 'User-Agent': 'python-zip-listing'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    f = RangeFile(URL, BYTES)
    zf = zipfile.ZipFile(f)
    names = {i.filename: i.file_size for i in zf.infolist()}
    sel = list(csv.DictReader(open(ROOT/'docs/handoff/artifacts/janek/boxquality_p0_20261124/selection_janek.csv', encoding='utf-8')))
    missing = [x['clean_relpath'] for x in sel if x['clean_relpath'] not in names]
    out = dict(url=URL, bytes=BYTES, members=len(names), bytes_downloaded_for_listing=f.fetched,
               train_images_members=sum(n.startswith('train_images/') for n in names),
               selected=len(sel), selected_missing=len(missing), examples_missing=missing[:5],
               selected_total_uncompressed_mib=sum(names.get(x['clean_relpath'], 0) for x in sel)/2**20,
               note='Byte-level SHA-256 of each image is checked in the notebook after extraction (train_ids.csv).')
    (ROOT/'docs/handoff/artifacts/janek/boxquality_p0_20261124/zip_listing_janek.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
