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

import asyncio
import fractions
import json
import queue
import socket
import time

import aiohttp
import av
import numpy as np
import pytest
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)

from dora_openarm_keyboard.web import (
    WebTeleopServer,
    _normalize_browser_key,
)


def test_normalize_printable():
    assert _normalize_browser_key("W") == "w"
    assert _normalize_browser_key(";") == ";"
    assert _normalize_browser_key("+") == "+"


def test_normalize_unusable():
    assert _normalize_browser_key("ArrowUp") is None
    assert _normalize_browser_key("Backspace") is None
    assert _normalize_browser_key("") is None
    assert _normalize_browser_key(None) is None
    assert _normalize_browser_key(3) is None


def test_normalize_teleop_controls():
    assert _normalize_browser_key("Shift") == "shift"
    assert _normalize_browser_key("Escape") == "escape"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _encode_jpeg() -> bytes:
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = 64
    codec.height = 48
    codec.pix_fmt = "yuvj420p"
    codec.time_base = fractions.Fraction(1, 30)
    image = np.full((48, 64, 3), 128, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    packets = codec.encode(frame.reformat(format="yuvj420p"))
    packets += codec.encode(None)
    return b"".join(bytes(packet) for packet in packets)


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.05)


def test_end_to_end():
    # One in-process WebRTC session against the server: the page HTML is
    # served, key events on the data channel come out of on_key, and pushed
    # JPEG frames come back as decoded video.
    events: queue.SimpleQueue = queue.SimpleQueue()
    port = _free_port()
    server = WebTeleopServer(
        on_key=lambda action, name: events.put((action, name)),
        host="127.0.0.1",
        port=port,
    )
    asyncio.run(_run_session(server, events, port))

    drained = []
    while True:
        try:
            drained.append(events.get_nowait())
        except queue.Empty:
            break
    assert ("press", "w") in drained
    assert ("release", "w") in drained


async def _run_session(server, events, port):
    await server.start()
    try:
        await _run_client(server, events, port)
    finally:
        await server.stop()


async def _run_client(server, events, port):
    jpeg = _encode_jpeg()
    received: dict = {}

    async with aiohttp.ClientSession() as session:
        response = await session.get(f"http://127.0.0.1:{port}/")
        page = await response.text()
        assert "teleop.js" in page

        response = await session.get(f"http://127.0.0.1:{port}/teleop.js")
        script = await response.text()
        assert "keydown" in script
        assert '"Shift"' in script
        assert '"Escape"' in script

        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            channel = pc.createDataChannel("keys")

            @pc.on("datachannel")
            def on_datachannel(dc):
                if dc.label == "help":

                    @dc.on("message")
                    def on_message(message):
                        received["help"] = message

            @pc.on("track")
            def on_track(track):
                async def read_one():
                    received["frame"] = await track.recv()

                asyncio.ensure_future(read_one())

            pc.addTransceiver("video", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())
            response = await session.post(
                f"http://127.0.0.1:{port}/offer",
                json={
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                },
            )
            answer = await response.json()
            await pc.setRemoteDescription(RTCSessionDescription(**answer))

            await _wait_for(lambda: channel.readyState == "open")
            channel.send(json.dumps({"type": "keydown", "key": "W"}))
            channel.send(json.dumps({"type": "keyup", "key": "W"}))
            channel.send(json.dumps({"type": "keydown", "key": "ArrowUp"}))
            channel.send("not json")
            await _wait_for(lambda: events.qsize() >= 2)

            deadline = time.monotonic() + 10.0
            while "frame" not in received:
                if time.monotonic() > deadline:
                    raise TimeoutError("no video frame arrived")
                server.push_jpeg(jpeg)
                await asyncio.sleep(0.05)
            assert received["frame"].width == 64
            assert received["frame"].height == 48

            await _wait_for(lambda: "help" in received)
            assert "Right arm" in received["help"]
        finally:
            await pc.close()


def test_oneshot_signaling():
    # WebRTC-only mode: no HTTP server. An offer is handed to negotiate_oneshot,
    # the answer comes back over a TCP socket the caller listens on, and once
    # the browser applies it the data channel carries keys to on_key as usual.
    events: queue.SimpleQueue = queue.SimpleQueue()
    server = WebTeleopServer(
        on_key=lambda action, name: events.put((action, name)),
        host="127.0.0.1",
        port=0,
    )
    asyncio.run(_run_oneshot(server, events))
    assert events.get_nowait() == ("press", "w")


async def _run_oneshot(server, events):
    answer: dict = {}

    async def handle_answer(reader, writer):
        answer["sdp"] = (await reader.read()).decode("utf-8")
        writer.close()

    tcp = await asyncio.start_server(handle_answer, "127.0.0.1", 0)
    host, port = tcp.sockets[0].getsockname()[:2]

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    try:
        async with tcp:
            channel = pc.createDataChannel("keys")
            pc.addTransceiver("video", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())

            # negotiate_oneshot blocks until the peer connects, so relaying the
            # answer to the browser has to happen concurrently, the way a real
            # signaling broker would.
            negotiate = asyncio.ensure_future(
                server.negotiate_oneshot(pc.localDescription.sdp, host, port, 10.0)
            )
            await _wait_for(lambda: "sdp" in answer)
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type="answer")
            )
            await negotiate

            await _wait_for(lambda: channel.readyState == "open")
            channel.send(json.dumps({"type": "keydown", "key": "W"}))
            await _wait_for(lambda: events.qsize() >= 1)
    finally:
        await pc.close()
        await server.stop()


def test_oneshot_disconnect():
    # Once the one browser of WebRTC-only mode goes away, the server stops
    # running so the node can exit: with no HTTP server, no other browser can
    # ever take its place.
    server = WebTeleopServer(on_key=lambda action, name: None, host="127.0.0.1", port=0)
    asyncio.run(_run_oneshot_disconnect(server))


async def _run_oneshot_disconnect(server):
    answer: dict = {}

    async def handle_answer(reader, writer):
        answer["sdp"] = (await reader.read()).decode("utf-8")
        writer.close()

    tcp = await asyncio.start_server(handle_answer, "127.0.0.1", 0)
    host, port = tcp.sockets[0].getsockname()[:2]

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    try:
        async with tcp:
            pc.createDataChannel("keys")
            pc.addTransceiver("video", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())

            negotiate = asyncio.ensure_future(
                server.negotiate_oneshot(pc.localDescription.sdp, host, port, 10.0)
            )
            await _wait_for(lambda: "sdp" in answer)
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type="answer")
            )
            await negotiate
            assert server.running

            await pc.close()
            await _wait_for(lambda: not server.running)
    finally:
        await pc.close()
        await server.stop()


def test_oneshot_connect_timeout():
    # The answer is sent but the browser never applies it, so the peer never
    # connects: negotiate_oneshot gives up after the timeout and raises.
    server = WebTeleopServer(on_key=lambda action, name: None, host="127.0.0.1", port=0)
    asyncio.run(_run_oneshot_timeout(server))


async def _run_oneshot_timeout(server):
    async def handle_answer(reader, writer):
        await reader.read()
        writer.close()

    tcp = await asyncio.start_server(handle_answer, "127.0.0.1", 0)
    host, port = tcp.sockets[0].getsockname()[:2]

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    try:
        async with tcp:
            pc.createDataChannel("keys")
            pc.addTransceiver("video", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())
            with pytest.raises(RuntimeError):
                await server.negotiate_oneshot(pc.localDescription.sdp, host, port, 0.5)
    finally:
        await pc.close()
        await server.stop()
