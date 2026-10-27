# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Browser teleoperation over WebRTC.

Serves a single-page client (``static/index.html`` and ``static/teleop.js``)
plus a ``POST /offer``
signaling endpoint, then talks WebRTC with the browser:

* key events arrive on a data channel labelled ``keys`` as JSON
  ``{"type": "keydown" | "keyup", "key": <KeyboardEvent.key>}`` and are handed
  to ``main`` through the ``on_key`` callback as
  ``("press" | "release", name)``, the shape the teleoperation core consumes;
* the key bindings go the other way on a data channel labelled ``help``: the
  node opens it and sends :data:`~.keymap.HELP_TEXT` once, so the page shows the
  current bindings even when a different host served it;
* MuJoCo camera frames flow the other way too: ``main`` pushes the JPEG payload
  of every ``image`` input via :meth:`WebTeleopServer.push_jpeg` and each
  connected browser receives them as a live video track.

For deployments where another service hosts the page and brokers signaling,
:meth:`WebTeleopServer.negotiate_oneshot` answers a single offer handed in at
startup and writes the answer to a TCP socket, with no HTTP server at all; if
the browser does not connect within a timeout it raises, so the node exits
rather than stranding.  Once that one browser disconnects no other can ever
take its place, so :attr:`WebTeleopServer.running` turns False and ``main``
exits.

Everything — the HTTP server, WebRTC, and the dora polling loop in ``main`` —
runs on one asyncio loop, so ``push_jpeg`` and the ``on_key`` callback are
plain same-loop calls and no locking is involved anywhere.

If a browser goes away mid-hold (tab closed, network drop), the keys it held
are released server-side, so a lost client can never leave the robot moving.
"""

import asyncio
import fractions
import json
import time
from collections.abc import Callable
from importlib import resources

import av
from aiohttp import web as aioweb
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import VideoStreamTrack

from .keymap import HELP_TEXT, QUIT_KEY, ROTATION_KEY

# RTP video clock; pts for outgoing frames are expressed in this rate.
_CLOCK_RATE = 90_000

OnKey = Callable[[str, str], None]


def _normalize_browser_key(key: object) -> str | None:
    """Map a browser ``KeyboardEvent.key`` value to its keymap name.

    Printable keys arrive as themselves (``"w"``, ``"+"``, ``";"``), already
    matching the keymap after lowercasing — including the shifted forms the
    browser reports while Shift is down (``"W"``).  ``Shift`` and ``Escape``
    are the special control keys.  Everything else — arrows and function
    keys — is not usable and maps to None.
    """
    if not isinstance(key, str) or not key:
        return None
    if len(key) == 1:
        return key.lower()
    special_keys = {
        "shift": ROTATION_KEY,
        "escape": QUIT_KEY,
    }
    special = special_keys.get(key.lower())
    if special is not None:
        return special
    return None


class FrameSource:
    """Latest-JPEG mailbox between the dora polling loop and the video tracks.

    Only the newest frame is kept: a slow encoder skips ahead instead of
    building a queue, which is the right trade for teleoperation video.
    """

    def __init__(self) -> None:
        """Start empty; the first frame arrives with the first publish."""
        self.jpeg: bytes | None = None
        self.version = 0
        self._event = asyncio.Event()

    def publish(self, jpeg: bytes) -> None:
        """Replace the current frame and wake every waiting track."""
        self.jpeg = jpeg
        self.version += 1
        self._event.set()

    async def wait_next(self, seen_version: int) -> tuple[bytes, int]:
        """Wait until a frame newer than ``seen_version`` exists, return it."""
        while self.version == seen_version:
            self._event.clear()
            await self._event.wait()
        assert self.jpeg is not None
        return self.jpeg, self.version


class JPEGVideoTrack(VideoStreamTrack):
    """Video track that decodes shared JPEG frames on demand.

    Each connection gets its own track (and its own MJPEG decoder, which is
    stateful), but all tracks read the same :class:`FrameSource`, so every
    browser sees the same camera at its own pace.
    """

    def __init__(self, source: FrameSource) -> None:
        """Attach to a frame source at version zero (before its first frame)."""
        super().__init__()
        self._source = source
        self._seen_version = 0
        self._decoder = av.CodecContext.create("mjpeg", "r")
        self._t0: float | None = None
        self._warned_decode = False

    async def recv(self) -> av.VideoFrame:
        """Return the next camera frame, timestamped against a local clock."""
        while True:
            jpeg, self._seen_version = await self._source.wait_next(self._seen_version)
            try:
                frames = self._decoder.decode(av.Packet(jpeg))
            except av.error.FFmpegError:
                if not self._warned_decode:
                    self._warned_decode = True
                    print(
                        "WARNING: the image input is not decodable JPEG",
                        flush=True,
                    )
                continue
            if frames:
                break
        frame = frames[-1]
        # Wall-clock pts instead of VideoStreamTrack's fixed 30 fps pacing:
        # frames should leave the moment MuJoCo renders them, whatever its rate.
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        frame.pts = int((now - self._t0) * _CLOCK_RATE)
        frame.time_base = fractions.Fraction(1, _CLOCK_RATE)
        return frame


class WebTeleopServer:
    """HTTP + WebRTC server bridging browsers to the teleoperation core."""

    def __init__(self, on_key: OnKey, host: str, port: int) -> None:
        """Prepare a server; nothing listens until :meth:`start`."""
        self._on_key = on_key
        self._host = host
        self._port = port
        self._source = FrameSource()
        self._pcs: set[RTCPeerConnection] = set()
        self._runner: aioweb.AppRunner | None = None
        self._running = True

    @property
    def running(self) -> bool:
        """True while the server can still serve a browser.

        In HTTP-server mode, where browsers come and go, this stays True
        until :meth:`stop`.  In WebRTC-only mode it also turns False once
        the single peer of :meth:`negotiate_oneshot` has failed or closed:
        no other browser can ever take its place, so the caller should exit.
        """
        return self._running

    async def start(self) -> None:
        """Start serving on the running loop; raise if the port cannot bind."""
        app = aioweb.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/teleop.js", self._handle_script)
        app.router.add_post("/offer", self._handle_offer)
        runner = aioweb.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            await aioweb.TCPSite(runner, self._host, self._port).start()
        except OSError as error:
            await runner.cleanup()
            raise RuntimeError(
                f"web server could not listen on {self._host}:{self._port}: {error}"
            ) from error
        self._runner = runner

    def push_jpeg(self, jpeg: bytes) -> None:
        """Hand a JPEG frame to the connected browsers.  Never blocks."""
        self._source.publish(jpeg)

    async def stop(self) -> None:
        """Close every connection and stop listening."""
        self._running = False
        for pc in list(self._pcs):
            await pc.close()
        self._pcs.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @staticmethod
    def _static_text(name: str) -> str:
        return (resources.files(__package__) / "static" / name).read_text(
            encoding="utf-8"
        )

    async def _handle_index(self, request: aioweb.Request) -> aioweb.Response:
        return aioweb.Response(
            text=self._static_text("index.html"), content_type="text/html"
        )

    async def _handle_script(self, request: aioweb.Request) -> aioweb.Response:
        return aioweb.Response(
            text=self._static_text("teleop.js"), content_type="text/javascript"
        )

    def _create_peer(self) -> RTCPeerConnection:
        """Build a peer connection wired to the teleoperation core.

        The browser opens the ``keys`` channel to send key events; we open a
        ``help`` channel the other way and push :data:`HELP_TEXT` down it once
        it is open, so the page shows the current bindings without any HTTP
        fetch — the only source of the help text is this node's keymap.
        """
        iceServers = [RTCIceServer(urls=["stun:stun.cloudflare.com:3478"])]
        pc = RTCPeerConnection(RTCConfiguration(iceServers=iceServers))
        self._pcs.add(pc)
        held: set[str] = set()

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            @channel.on("message")
            def on_message(message) -> None:
                self._handle_key_message(message, held)

        help_channel = pc.createDataChannel("help")

        @help_channel.on("open")
        def on_help_open() -> None:
            help_channel.send(HELP_TEXT)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in ("failed", "closed"):
                # The browser is gone; whatever it held must not keep moving.
                for name in held:
                    self._on_key("release", name)
                if held:
                    print(
                        f"browser left with keys held; released {sorted(held)}",
                        flush=True,
                    )
                else:
                    print("browser disconnected", flush=True)
                held.clear()
                self._pcs.discard(pc)
                await pc.close()

        pc.addTrack(JPEGVideoTrack(self._source))
        return pc

    @staticmethod
    async def _answer(pc: RTCPeerConnection, offer: RTCSessionDescription) -> dict:
        """Consume an offer and return the SDP answer as a ``{sdp, type}`` dict.

        aiortc's ``setLocalDescription`` waits for ICE gathering to finish, so
        the returned SDP already carries every candidate: signaling is a single
        offer/answer round with no trickle.
        """
        await pc.setRemoteDescription(offer)
        await pc.setLocalDescription(await pc.createAnswer())
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def _handle_offer(self, request: aioweb.Request) -> aioweb.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = self._create_peer()
        return aioweb.json_response(await self._answer(pc, offer))

    async def negotiate_oneshot(
        self,
        offer_sdp: str,
        answer_host: str,
        answer_port: int,
        connect_timeout: float = 60.0,
    ) -> None:
        """Answer a single offer handed in at startup, with no HTTP server.

        The offer arrives out of band (a command-line argument or environment
        variable) as the bare SDP -- its type is always ``offer`` -- and the
        answer goes back the same way, the bare answer SDP written to the TCP
        socket the caller is listening on at ``answer_host``/``answer_port``.
        This is the WebRTC-only mode: another service hosts the page and brokers
        signaling, and this node just runs the peer for its lifetime.

        After sending the answer, waits up to ``connect_timeout`` seconds for the
        peer to connect. If it never does -- the browser never applied the
        answer, or the media path never came up -- the peer is closed and this
        raises :class:`RuntimeError`, so a stranded one-shot node exits instead
        of holding a dead connection forever.

        Once connected, if the peer later fails or closes -- the tab closed,
        the network dropped -- :attr:`running` turns False, so the caller can
        exit: with no HTTP server, no other browser can ever replace the one
        that left.
        """
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        pc = self._create_peer()

        established: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        @pc.on("connectionstatechange")
        def on_connectionstatechange() -> None:
            if pc.connectionState == "connected":
                if not established.done():
                    established.set_result(True)
            elif pc.connectionState in ("failed", "closed"):
                if not established.done():
                    established.set_result(False)
                self._running = False

        answer = await self._answer(pc, offer)

        _reader, writer = await asyncio.open_connection(answer_host, answer_port)
        writer.write(answer["sdp"].encode("utf-8"))
        await writer.drain()
        writer.write_eof()
        writer.close()
        await writer.wait_closed()

        try:
            connected = await asyncio.wait_for(established, connect_timeout)
        except TimeoutError:
            connected = False
        if not connected:
            await pc.close()
            raise RuntimeError(
                f"no WebRTC connection within {connect_timeout:g}s of the answer"
            )

    def _handle_key_message(self, message: object, held: set[str]) -> None:
        if not isinstance(message, str):
            return
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        name = _normalize_browser_key(event.get("key"))
        if name is None:
            return
        kind = event.get("type")
        if kind == "keydown":
            held.add(name)
            self._on_key("press", name)
        elif kind == "keyup":
            held.discard(name)
            self._on_key("release", name)
