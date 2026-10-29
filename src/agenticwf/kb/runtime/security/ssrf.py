"""
SSRF-safe HTTP fetch for URL import and refresh (docs/ingestion.md, SSRF-safe fetch).

    await fetch(url, max_bytes=…, timeout=…) -> Fetched(url, status, content_type, body)

* only http and https;
* the host is resolved once and every address must be public (not private, loopback,
  link-local, multicast, reserved, unspecified), unless the host or the address is in
  `[security] ssrf_whitelist` (host names and CIDRs, comma or space separated);
* the connection goes to the checked address (DNS pinned, so a second lookup cannot change it),
  while TLS still verifies the certificate against the host name (SNI);
* redirects are followed by hand, at most 5, each one checked again;
* the body is read up to `max_bytes`; a larger response is an error.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from agenticwf.kb.core.common.config import config_get, config_has_option

MAX_REDIRECTS = 5
USER_AGENT = "agenticwf-kb/0.1 (+https://github.com/InnovateAILab)"


class FetchError(Exception):
    pass


@dataclass
class Fetched:
    url: str
    status: int
    content_type: str
    body: bytes
    retry_after: str = ""        # the Retry-After header, for a 429 or 503


def whitelist() -> tuple[set[str], list[ipaddress._BaseNetwork]]:
    raw = config_get("security", "ssrf_whitelist") if config_has_option("security", "ssrf_whitelist") else ""
    hosts, nets = set(), []
    for item in raw.replace(",", " ").split():
        try:
            nets.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            hosts.add(item.lower())
    return hosts, nets


def _public(address: ipaddress._BaseAddress) -> bool:
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast
                or address.is_reserved or address.is_unspecified)


async def resolve_checked(host: str, port: int) -> str:
    """The address to connect to, after checking every address the host resolves to."""
    hosts, nets = whitelist()
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
        infos = [literal]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            resolved = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise FetchError(f"cannot resolve {host}") from exc
        infos = [ipaddress.ip_address(info[4][0].split("%", 1)[0]) for info in resolved]
    if not infos:
        raise FetchError(f"cannot resolve {host}")
    allowed_host = host.lower() in hosts
    for address in infos:
        if not (allowed_host or _public(address) or any(address in net for net in nets)):
            raise FetchError(f"refusing to fetch {host}: {address} is not a public address "
                             f"(add it to [security] ssrf_whitelist if it should be allowed)")
    return str(infos[0])


async def fetch(url: str, max_bytes: int = 20 * 1024 * 1024, timeout: float = 30.0) -> Fetched:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(current)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise FetchError(f"only http and https URLs can be imported: {current}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        address = await resolve_checked(parts.hostname, port)
        host_for_url = f"[{address}]" if ":" in address else address
        netloc = f"{host_for_url}:{port}"
        pinned = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
        host_header = parts.hostname + (f":{parts.port}" if parts.port else "")
        extensions = {"sni_hostname": parts.hostname} if parts.scheme == "https" else {}
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                request = client.build_request("GET", pinned, headers={"Host": host_header, "User-Agent": USER_AGENT},
                                               extensions=extensions)
                response = await client.send(request, stream=True)
                try:
                    if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
                        current = urljoin(current, response.headers["location"])
                        continue
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > max_bytes:
                            raise FetchError(f"{url} is larger than {max_bytes // (1024 * 1024)} MB")
                    return Fetched(url=current, status=response.status_code,
                                   content_type=response.headers.get("content-type", "").split(";")[0].strip().lower(),
                                   body=bytes(body), retry_after=response.headers.get("retry-after", ""))
                finally:
                    await response.aclose()
        except httpx.HTTPError as exc:
            raise FetchError(f"fetching {current} failed: {exc}") from exc
    raise FetchError(f"too many redirects from {url}")
