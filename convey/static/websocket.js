// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

/**
 * Callosum WebSocket Bridge
 *
 * Connects to /ws/events and broadcasts Callosum events to registered listeners.
 * Provides window.appEvents API for subscribing to events by tract.
 */
(function(){
  const listeners = {};  // Keyed by tract: 'cortex', 'task', 'indexer', etc.
  let ws;
  let retry = 1000;
  let statusIcon = null;

  // Connection metrics
  let connectedAt = null;
  let lastMessageAt = null;
  let isConnected = false;

  // Update status icon (if present)
  function updateStatusIcon(connected) {
    if (!statusIcon) {
      statusIcon = document.querySelector('.facet-bar .status-icon');
    }

    if (statusIcon) {
      statusIcon.textContent = connected ? '🟢' : '🔴';
      statusIcon.setAttribute('title', connected ? 'Connected' : 'Disconnected');
    }

    isConnected = connected;
  }

  // Connect to WebSocket
  function connect(){
    ws = new WebSocket(`ws://${location.host}/ws/events`);

    ws.onopen = () => {
      connectedAt = Date.now();
      lastMessageAt = null;
      updateStatusIcon(true);
      retry = 1000;
      console.debug('[WebSocket] Connected to /ws/events');
    };

    ws.onclose = () => {
      connectedAt = null;
      lastMessageAt = null;
      updateStatusIcon(false);
      retry = Math.min(retry * 1.5, 15000);
      console.debug(`[WebSocket] Disconnected, reconnecting in ${retry}ms`);
      setTimeout(connect, retry);
    };

    ws.onmessage = e => {
      lastMessageAt = Date.now();

      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch(err) {
        console.warn('[WebSocket] Failed to parse message:', err);
        return;
      }

      const tract = msg.tract;

      // Call tract-specific listeners
      if(tract && listeners[tract]){
        listeners[tract].forEach(fn => {
          try {
            fn(msg);
          } catch(err) {
            console.error(`[WebSocket] Error in ${tract} listener:`, err);
          }
        });
      }

      // Call wildcard listeners
      if(listeners['*']){
        listeners['*'].forEach(fn => {
          try {
            fn(msg);
          } catch(err) {
            console.error('[WebSocket] Error in wildcard listener:', err);
          }
        });
      }
    };

    ws.onerror = (err) => {
      console.error('[WebSocket] Error:', err);
    };
  }

  // Expose global API
  window.appEvents = {
    /**
     * Listen for events from a specific tract or all events.
     *
     * @param {string} tract - Tract name ('cortex', 'observe', 'indexer', etc.) or '*' for all
     * @param {function} fn - Callback function that receives the event object
     * @returns {function} Cleanup function to remove the listener
     *
     * @example
     * // Listen to cortex events
     * const cleanup = window.appEvents.listen('cortex', (msg) => {
     *   console.log('Cortex event:', msg);
     * });
     *
     * // Later, remove listener
     * cleanup();
     *
     * @example
     * // Listen to all events
     * window.appEvents.listen('*', (msg) => {
     *   console.log('Event:', msg.tract, msg.event);
     * });
     */
    listen(tract, fn){
      if(!listeners[tract]) listeners[tract] = [];
      listeners[tract].push(fn);
      // Return cleanup function
      return () => this.unlisten(tract, fn);
    },

    /**
     * Remove a specific listener for a tract.
     *
     * @param {string} tract - Tract name or '*'
     * @param {function} fn - The listener function to remove
     */
    unlisten(tract, fn){
      if(listeners[tract]){
        listeners[tract] = listeners[tract].filter(f => f !== fn);
      }
    },

    /**
     * Get connection metrics.
     *
     * @returns {object} Object with connection status and timing info
     */
    getMetrics(){
      const now = Date.now();
      return {
        connected: isConnected,
        uptimeMs: connectedAt ? now - connectedAt : 0,
        lastMessageMs: lastMessageAt ? now - lastMessageAt : null,
        lastMessageAt: lastMessageAt,
        connectedAt: connectedAt
      };
    }
  };

  // Built-in tract: forward notification events to the in-app notification UI
  listeners['notification'] = [function(msg) {
    window.AppServices?.notifications?.show(msg);
  }];

  // Auto-connect when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connect);
  } else {
    // DOM already loaded, connect immediately
    connect();
  }
})();
