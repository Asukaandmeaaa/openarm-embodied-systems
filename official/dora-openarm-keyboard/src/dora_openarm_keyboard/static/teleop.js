// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const statusElement = document.getElementById("status");
const heldElement = document.getElementById("held");
const helpElement = document.getElementById("help");
const video = document.getElementById("video");

// The key bindings arrive over a WebRTC "help" data channel the node opens,
// not over HTTP: the help text lives in the node's keymap, and this page may
// be served by a different host that has no copy of it.

let channel = null;
const held = new Set();

function setStatus(text) {
  statusElement.textContent = text;
}

function showHeld() {
  heldElement.textContent = held.size
    ? "held: " + [...held].sort().join(" ")
    : "";
}

function send(type, key) {
  if (channel && channel.readyState === "open") {
    channel.send(JSON.stringify({ type, key }));
  }
}

// Only forward keys the node can use: printable characters and the special
// control keys used by the teleop mapping. Modifier chords (Cmd+R, Ctrl+W,
// …) pass through untouched.
//
// Printable keys are lowercased here because Shift — the rotation modifier —
// changes what the browser reports ("w" becomes "W"): a key pressed before
// Shift and released after it would otherwise never leave `held`. The node
// lowercases too, so this only keeps the on-screen list honest.
function usableKey(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return null;
  if (event.key.length === 1) return event.key.toLowerCase();
  if (["Shift", "Escape"].includes(event.key)) return event.key;
  return null;
}

window.addEventListener("keydown", (event) => {
  const key = usableKey(event);
  if (key === null) return;
  event.preventDefault();
  if (event.repeat) return;
  held.add(key);
  showHeld();
  send("keydown", key);
});

window.addEventListener("keyup", (event) => {
  const key = usableKey(event);
  if (key === null) return;
  event.preventDefault();
  held.delete(key);
  showHeld();
  send("keyup", key);
});

// Losing focus loses keyup events, so release everything: an unfocused page
// must never keep the robot moving.
function releaseAll() {
  for (const key of held) send("keyup", key);
  held.clear();
  showHeld();
}
window.addEventListener("blur", releaseAll);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseAll();
});
window.addEventListener("pagehide", releaseAll);

async function connect() {
  const configuration = {
    iceServers: [
      {
        urls: ["stun:stun.cloudflare.com:3478"],
      },
    ],
  };
  const pc = new RTCPeerConnection(configuration);

  channel = pc.createDataChannel("keys");
  channel.onopen = () =>
    setStatus("connected — click the page, then hold keys to move");
  channel.onclose = () => setStatus("disconnected — reload to reconnect");

  pc.ontrack = (event) => {
    video.srcObject = event.streams[0];
  };
  pc.addTransceiver("video", { direction: "recvonly" });

  // The node opens a "help" channel and sends the key bindings once.
  pc.ondatachannel = (event) => {
    if (event.channel.label === "help") {
      event.channel.onmessage = (message) => {
        helpElement.textContent = message.data;
      };
    }
  };

  await pc.setLocalDescription(await pc.createOffer());
  // Non-trickle ICE: signaling is a single POST to /offer, and the server
  // only learns candidates from the SDP it receives — there is no endpoint
  // to send candidates one by one afterwards. setLocalDescription() resolves
  // before gathering finishes, so wait until every candidate has been added
  // to localDescription.sdp before sending it. The browser has no
  // promise-based API for this; bridge the icegatheringstatechange event
  // into an awaitable, checking the current state first in case gathering
  // already finished before the listener was attached.
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
      return;
    }
    pc.addEventListener("icegatheringstatechange", () => {
      if (pc.iceGatheringState === "complete") resolve();
    });
  });

  const response = await fetch("offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  if (!response.ok) throw new Error("signaling failed: " + response.status);
  const description = await response.json();
  await pc.setRemoteDescription(description);
}

connect().catch((error) => setStatus("connection failed: " + error.message));
