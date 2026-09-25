"""Doublures de test partagées (aucun réseau) : faux bucket S3 de data.binance.vision."""

from __future__ import annotations

import hashlib
import urllib.parse

NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def s3_xml(
    prefix: str,
    keys: list[tuple[str, int]],
    subdirs: list[str],
    truncated: bool,
    next_marker: str | None = None,
) -> bytes:
    parts = [
        f'<ListBucketResult xmlns="{NS}"><Prefix>{prefix}</Prefix>',
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>",
    ]
    if next_marker:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts += [f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in keys]
    parts += [f"<CommonPrefixes><Prefix>{d}</Prefix></CommonPrefixes>" for d in subdirs]
    return ("".join(parts) + "</ListBucketResult>").encode()


class FakeS3:
    """Bucket factice : ``files`` (clé → taille), pages de ``page_size`` entrées."""

    def __init__(self, files: dict[str, int], page_size: int = 1000) -> None:
        self.files, self.page_size, self.calls = files, page_size, 0

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True))
        prefix, marker, delim = q["prefix"], q.get("marker", ""), "delimiter" in q
        keys = sorted(k for k in self.files if k.startswith(prefix))
        if delim:
            entries = sorted(
                {
                    prefix + k[len(prefix) :].split("/")[0] + "/" if "/" in k[len(prefix) :] else k
                    for k in keys
                }
            )
        else:
            entries = keys
        entries = [e for e in entries if e > marker]
        page, rest = entries[: self.page_size], entries[self.page_size :]
        subdirs = [e for e in page if e.endswith("/")]
        files = [(e, self.files[e]) for e in page if not e.endswith("/")]
        nxt = page[-1] if rest and delim else None
        return s3_xml(prefix, files, subdirs, bool(rest), nxt)


class FakeVision:
    """Faux data.binance.vision complet : listage S3 (``list_url``), archives et ``.CHECKSUM``
    (``base_url``). ``content`` : clé → octets ; ``bad_checksum`` : clés à empreinte fausse ;
    ``failing`` : clés dont le téléchargement lève l'exception donnée."""

    def __init__(
        self,
        content: dict[str, bytes],
        *,
        list_url: str = "https://s3.example/bucket",
        base_url: str = "https://vision.example",
        page_size: int = 1000,
        bad_checksum: set[str] | None = None,
        failing: dict[str, Exception] | None = None,
    ) -> None:
        self.content, self.list_url, self.base_url = content, list_url, base_url
        self.bad_checksum, self.failing = bad_checksum or set(), failing or {}
        self.s3 = FakeS3({k: len(v) for k, v in content.items()}, page_size)
        self.downloads: list[str] = []

    def __call__(self, url: str) -> bytes:
        if url.startswith(self.list_url + "?"):
            return self.s3(url)
        key = url.removeprefix(self.base_url + "/")
        if key.endswith(".CHECKSUM"):
            zip_key = key.removesuffix(".CHECKSUM")
            sha = hashlib.sha256(self.content[zip_key]).hexdigest()
            if zip_key in self.bad_checksum:
                sha = "0" * 64
            return f"{sha}  {zip_key.rsplit('/', 1)[-1]}\n".encode()
        if key in self.failing:
            raise self.failing[key]
        self.downloads.append(key)
        return self.content[key]
