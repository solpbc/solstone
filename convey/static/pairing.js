// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

(function () {
  function $(id) {
    return document.getElementById(id);
  }

  const generateButton = $("pairing-generate");
  const feedback = $("pairing-feedback");
  const mintedSection = $("pairing-minted");
  const qrContainer = $("pairing-qr");
  const pairingUrl = $("pairing-url");
  const countdown = $("pairing-countdown");
  const devicesEmpty = $("pairing-devices-empty");
  const devicesList = $("pairing-devices-list");

  let countdownTimer = null;
  let pollTimer = null;

  async function fetchJson(url, options) {
    const response = await fetch(
      url,
      Object.assign({ credentials: "same-origin" }, options || {}),
    );

    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }

    if (!response.ok) {
      const message = payload && payload.error ? payload.error : "Request failed";
      throw new Error(message);
    }

    return payload || {};
  }

  function setFeedback(message, isError) {
    if (!feedback) {
      return;
    }
    feedback.textContent = message || "";
    feedback.dataset.state = isError ? "error" : "info";
  }

  function renderQr(data) {
    if (!qrContainer) {
      return;
    }
    qrContainer.innerHTML = "";

    if (typeof window.qrcode !== "function") {
      qrContainer.textContent = "QR unavailable";
      return;
    }

    const qr = window.qrcode(0, "M");
    qr.addData(data);
    qr.make();
    qrContainer.innerHTML = qr.createSvgTag({
      cellSize: 6,
      margin: 0,
      scalable: true,
      title: "Phone pairing code",
      alt: "Phone pairing code",
    });
  }

  function renderCountdown(expiresAt) {
    if (!countdown) {
      return;
    }

    function update() {
      const remaining = Math.max(0, expiresAt - Math.floor(Date.now() / 1000));
      countdown.textContent =
        remaining > 0 ? `Expires in ${remaining}s` : "Expired";
      if (remaining === 0 && countdownTimer !== null) {
        window.clearInterval(countdownTimer);
        countdownTimer = null;
      }
    }

    if (countdownTimer !== null) {
      window.clearInterval(countdownTimer);
    }
    update();
    countdownTimer = window.setInterval(update, 1000);
  }

  function renderDevices(devices) {
    if (!devicesList || !devicesEmpty) {
      return;
    }

    devicesList.innerHTML = "";
    const rows = Array.isArray(devices) ? devices : [];
    devicesEmpty.hidden = rows.length > 0;

    rows.forEach((device) => {
      const row = document.createElement("article");
      row.className = "pairing-device";

      const meta = document.createElement("div");
      meta.className = "pairing-device-meta";

      const name = document.createElement("h3");
      name.textContent = device.name || "Unnamed phone";
      meta.appendChild(name);

      const details = document.createElement("p");
      details.className = "pairing-device-details";
      details.textContent = [
        device.platform || "unknown",
        `paired ${device.paired_at || "unknown"}`,
        `last seen ${device.last_seen_at || "never"}`,
      ].join(" · ");
      meta.appendChild(details);

      const button = document.createElement("button");
      button.type = "button";
      button.className = "pairing-unpair";
      button.textContent = "Unpair";
      button.addEventListener("click", async function () {
        button.disabled = true;
        try {
          await fetchJson(`/api/pairing/devices/${encodeURIComponent(device.id)}`, {
            method: "DELETE",
          });
          setFeedback("Phone removed.", false);
          await refreshDevices();
        } catch (error) {
          setFeedback(error.message, true);
          button.disabled = false;
        }
      });

      row.appendChild(meta);
      row.appendChild(button);
      devicesList.appendChild(row);
    });
  }

  async function refreshDevices() {
    try {
      const payload = await fetchJson("/api/pairing/devices");
      renderDevices(payload.devices);
    } catch (error) {
      setFeedback(error.message, true);
    }
  }

  async function mintPairingCode() {
    if (!generateButton) {
      return;
    }
    generateButton.disabled = true;
    setFeedback("", false);

    try {
      const payload = await fetchJson("/api/pairing/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });

      if (mintedSection) {
        mintedSection.hidden = false;
      }
      if (pairingUrl) {
        pairingUrl.textContent = payload.pairing_url || "";
      }
      renderQr(payload.qr_data || "");
      renderCountdown(payload.expires_at || 0);
    } catch (error) {
      setFeedback(error.message, true);
    } finally {
      generateButton.disabled = false;
    }
  }

  if (generateButton) {
    generateButton.addEventListener("click", mintPairingCode);
  }

  refreshDevices();
  pollTimer = window.setInterval(refreshDevices, 5000);
  window.addEventListener("beforeunload", function () {
    if (countdownTimer !== null) {
      window.clearInterval(countdownTimer);
    }
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
    }
  });
})();
