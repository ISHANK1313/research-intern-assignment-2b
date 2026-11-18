#!/usr/bin/env python
"""
Download and extract the Toronto Emotional Speech Set (TESS) used by this assignment.

The raw audio is NEVER committed to this repository. This script fetches it into a
local, git-ignored directory so that every pipeline can be reproduced from scratch.

Usage
-----
    python scripts/download_tess.py                  # -> ./data/tess
    python scripts/download_tess.py --data-dir D:/tess_data

The Kaggle endpoint used here is public and requires no account or API key.
If the anonymous download ever stops working you can instead download the archive
in a browser and pass it with --zip.

Safety
------
Server-side URL fetching is restricted to http/https on public hosts: loopback,
private, link-local and reserved addresses are rejected before any request is
made (see `validate_public_url`).
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

KAGGLE_SLUG = "ejlok1/toronto-emotional-speech-set-tess"
URLS = [
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}",
]
ARCHIVE_NAME = "tess.zip"
USER_AGENT = "Mozilla/5.0 (compatible; tess-downloader/1.0)"


def validate_public_url(url: str) -> str:
    """Return `url` if it is a safe public http/https URL, else raise ValueError.

    Rejects non-http(s) schemes and any host that resolves to a loopback,
    private, link-local, multicast or otherwise reserved address.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no host: {url!r}")

    # Resolve every address the host maps to and check each one.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host {host!r}: {exc}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"refusing non-public address for host {host!r}: {ip}")
    return url


def _download(url: str, dest: Path) -> None:
    validate_public_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r  {done/1e6:7.1f} / {total/1e6:.1f} MB  ({pct:5.1f}%)",
                      end="", flush=True)
    print()


def _safe_extract(archive: Path, dest: Path) -> int:
    """Extract `archive` into `dest`, refusing entries that escape `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise ValueError(f"unsafe zip entry escapes target dir: {member.filename!r}")
            zf.extract(member, dest)
            n += 1
    return n


def count_wavs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.wav"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/tess",
                    help="directory the dataset is extracted into (git-ignored)")
    ap.add_argument("--zip", dest="zip_path", default=None,
                    help="use an already-downloaded archive instead of fetching one")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the downloaded archive after extraction")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(args.zip_path).expanduser().resolve() if args.zip_path \
        else data_dir / ARCHIVE_NAME

    if args.zip_path:
        if not archive.is_file():
            print(f"error: --zip file not found: {archive}", file=sys.stderr)
            return 2
        print(f"Using provided archive: {archive}")
    elif archive.is_file() and archive.stat().st_size > 1 << 20:
        print(f"Archive already present, skipping download: {archive}")
    else:
        last_err: Exception | None = None
        for url in URLS:
            try:
                print(f"Downloading {url}")
                _download(url, archive)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 - report and try the next mirror
                last_err = exc
                print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if last_err is not None:
            print("\nAll mirrors failed. Download TESS manually and re-run with "
                  f"--zip <path-to-archive>.", file=sys.stderr)
            return 1

    print(f"Extracting into {data_dir} ...")
    n = _safe_extract(archive, data_dir)
    print(f"Extracted {n} archive entries.")

    if not args.keep_zip and not args.zip_path:
        try:
            archive.unlink()
            print("Removed archive.")
        except OSError:
            pass

    wavs = count_wavs(data_dir)
    print(f"Found {wavs} .wav files under {data_dir}")
    if wavs == 0:
        print("warning: no .wav files found - check the archive layout.", file=sys.stderr)
        return 1
    print("Done. Pass this directory to the pipelines with --data-dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
