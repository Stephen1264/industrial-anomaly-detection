"""Verify and unpack the group's three published releases into separate directories."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

LOCK = json.loads(Path(__file__).with_name("inputs.json").read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(directory, parts):
    directory.mkdir(parents=True, exist_ok=True)
    for part in parts:
        spec = LOCK["inputs"][part]
        path = directory / spec["filename"]
        target = directory / part
        marker = target / ".verified-release-sha256"
        if marker.is_file() and marker.read_text().strip() == spec["sha256"]:
            continue
        if not path.is_file():
            url = "https://github.com/{}/releases/download/{}/{}".format(
                LOCK["repository"], spec["tag"], spec["filename"]
            )
            tmp = path.with_suffix(".download")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "3", "--output", str(tmp), url], check=True)
            if digest(tmp) != spec["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {tmp}; archive was not installed")
            os.replace(tmp, path)
        elif digest(path) != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}; replace it with the published release")
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                relative = PurePosixPath(item.filename.replace(chr(92), "/"))
                if relative.is_absolute() or ".." in relative.parts or any(":" in s for s in relative.parts):
                    raise ValueError(f"Invalid archive member {item.filename!r}")
                if not relative.parts or "__MACOSX" in relative.parts or relative.name.startswith("._"):
                    continue
                dest = target.joinpath(*relative.parts)
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
        marker.write_text(spec["sha256"] + "\n", encoding="ascii")
        print(f"{part}: {spec['tag']} verified and extracted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parts", nargs="+", choices=tuple(LOCK["inputs"]), default=list(LOCK["inputs"]))
    args = parser.parse_args()
    prepare(args.directory, args.parts)
