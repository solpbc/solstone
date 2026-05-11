// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

/**
 * Callosum SSE Bridge
 *
 * Connects to /sse/events and broadcasts Callosum events to registered listeners.
 * Provides window.appEvents API for subscribing to events by tract.
 */
(function(){
  const listeners = {};
  const parseErrorHandlers = new Set();
  const connectionStateHandlers = new Set();
  let eventSource;
  let statusIcon = null;

  // Connection metrics
  let connectedAt = null;
  let lastMessageAt = null;
  let connectionState = 'disconnected';
  let disconnectTimerId = null;
  let disconnectCardId = null;

  function getTractListeners(tract) {
    if (!listeners[tract]) {
      listeners[tract] = [];
    }
    return listeners[tract];
  }

  function notifyParseError(error, rawPayload) {
    parseErrorHandlers.forEach(handler => {
      try {
        handler(error, rawPayload);
      } catch (handlerError) {
        if (typeof window.logError === 'function') {
          window.logError(handlerError, { context: 'sse-parse-handler' });
        }
      }
    });

    if (typeof window.logError === 'function') {
      window.logError(error, { context: 'sse-parse' });
    }
  }

  function notifyConnectionState() {
    const payload = { connected: connectionState === 'connected', state: connectionState };
    connectionStateHandlers.forEach(handler => {
      try {
        handler(payload);
      } catch (handlerError) {
        if (typeof window.logError === 'function') {
          window.logError(handlerError, { context: 'sse-connection-handler' });
        }
      }
    });
  }

  function createPendingController(options) {
    const pending = new Map();
    const hasTimeout = Number.isFinite(options.timeout) && options.timeout > 0;
    const onTimeout = typeof options.onTimeout === 'function' ? options.onTimeout : null;

    return {
      track(correlationId) {
        if (!hasTimeout || !onTimeout || correlationId == null) {
          return correlationId;
        }
        this.clear(correlationId);
        const timeoutId = window.setTimeout(() => {
          pending.delete(correlationId);
          onTimeout(correlationId);
        }, options.timeout);
        pending.set(correlationId, timeoutId);
        return correlationId;
      },

      clear(correlationId) {
        if (correlationId == null) {
          return;
        }
        const timeoutId = pending.get(correlationId);
        if (timeoutId) {
          window.clearTimeout(timeoutId);
          pending.delete(correlationId);
        }
      },

      clearAll() {
        pending.forEach(timeoutId => window.clearTimeout(timeoutId));
        pending.clear();
      }
    };
  }

  function getCorrelationId(msg, correlationKey) {
    if (!msg) {
      return undefined;
    }
    if (typeof correlationKey === 'function') {
      return correlationKey(msg);
    }
    return msg[correlationKey || 'use_id'];
  }

  function validateSchema(msg, schema) {
    if (!schema) {
      return;
    }
    if (Array.isArray(schema)) {
      const missing = schema.filter(key => msg == null || msg[key] === undefined);
      if (missing.length > 0) {
        throw new Error(`Missing required SSE field(s): ${missing.join(', ')}`);
      }
      return;
    }
    if (typeof schema === 'function') {
      const result = schema(msg);
      if (result === false) {
        throw new Error('SSE schema validation failed');
      }
    }
  }

  function createListenerRecord(fn, options) {
    return {
      fn,
      options,
      pending: createPendingController(options)
    };
  }

  function addListenerRecord(tract, record) {
    getTractListeners(tract).push(record);
  }

  function removeListenerRecord(tract, record) {
    if (!listeners[tract]) {
      return;
    }
    record.pending.clearAll();
    listeners[tract] = listeners[tract].filter(candidate => candidate !== record);
  }

  function dispatchToRecords(tract, msg) {
    const records = listeners[tract];
    if (!records || records.length === 0) {
      return;
    }

    records.slice().forEach(record => {
      try {
        const correlationId = getCorrelationId(msg, record.options.correlationKey);
        if (correlationId != null) {
          record.pending.clear(correlationId);
        }

        validateSchema(msg, record.options.schema);
        record.fn(msg);
      } catch (err) {
        if (record.options.schema) {
          if (typeof record.options.onDrop === 'function') {
            try {
              record.options.onDrop(msg, err);
            } catch (dropError) {
              if (typeof window.logError === 'function') {
                window.logError(dropError, { context: 'sse-drop' });
              }
            }
          }
          notifyParseError(err, msg);
          return;
        }

        console.error(`[SSE] Error in ${tract} listener:`, err);
      }
    });
  }

  function updateStatusIcon(state) {
    if (!statusIcon) {
      statusIcon = document.querySelector('.facet-bar .status-icon');
    }

    if (statusIcon) {
      const badge = statusIcon.querySelector('#quiet-notif-badge');
      const svgs = {
        connected: '<svg class="status-indicator" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="#10b981"/></svg>',
        connecting: '<svg class="status-indicator status-indicator--connecting" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="24 8"/></svg>',
        disconnected: '<svg class="status-indicator" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M8 2 L14 13 L2 13 Z" fill="#ef4444"/></svg>'
      };
      const labels = {
        connected: 'connected',
        connecting: 'connecting',
        disconnected: 'disconnected'
      };

      statusIcon.innerHTML = svgs[state] || svgs.disconnected;
      if (badge) {
        statusIcon.appendChild(badge);
      }
      statusIcon.setAttribute('title', labels[state] || state);
    }

    const previousState = connectionState;
    connectionState = state;
    if (previousState !== state) {
      notifyConnectionState();
    }
    window.updateStatusLabel?.();
  }

  function connect() {
    updateStatusIcon('connecting');
    eventSource = new EventSource('/sse/events');

    eventSource.onopen = () => {
      connectedAt = Date.now();
      updateStatusIcon('connected');

      if (disconnectTimerId) {
        clearTimeout(disconnectTimerId);
        disconnectTimerId = null;
      }

      if (disconnectCardId !== null) {
        window.AppServices?.notifications?.dismiss(disconnectCardId);
        const reconnectedId = window.AppServices?.notifications?.show({
          app: 'system',
          icon: '✓',
          title: 'reconnected',
          message: 'all features restored',
          dismissible: true
        });
        if (reconnectedId != null) {
          setTimeout(() => window.AppServices?.notifications?.dismiss(reconnectedId), 3000);
        }
        disconnectCardId = null;
      }

      console.debug('[SSE] Connected to /sse/events');
    };

    eventSource.onerror = err => {
      connectedAt = null;
      updateStatusIcon('disconnected');

      if (!disconnectTimerId && disconnectCardId === null) {
        disconnectTimerId = setTimeout(() => {
          disconnectTimerId = null;
          const id = window.AppServices?.notifications?.show({
            app: 'system',
            icon: '⚠️',
            title: 'connection lost',
            message: 'reconnecting — some features may be delayed',
            dismissible: false
          });
          if (id != null) {
            disconnectCardId = id;
          }
        }, 5000);
      }

      console.error('[SSE] Error:', err);
    };

    eventSource.onmessage = event => {
      lastMessageAt = Date.now();

      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (err) {
        console.warn('[SSE] Failed to parse message:', err);
        notifyParseError(err, event.data);
        return;
      }

      const tract = msg.tract;
      if (tract) {
        dispatchToRecords(tract, msg);
      }
      dispatchToRecords('*', msg);
    };
  }

  window.appEvents = {
    /**
     * Listen for events from a specific tract or all events.
     *
     * @param {string} tract - Tract name ('cortex', 'observe', 'indexer', etc.) or '*' for all
     * @param {function|object} optionsOrFn - Callback or options object
     * @param {function} [fn] - Callback when using the `(tract, options, fn)` overload
     * @returns {function} Cleanup function with `.pending.track(correlationId)` and `.pending.clear(correlationId)`
     *
     * @example
     * const cleanup = window.appEvents.listen('importer', {
     *   schema: ['event', 'use_id'],
     *   timeout: 15000,
     *   onTimeout(useId) {
     *     console.warn('Importer request timed out:', useId);
     *   }
     * }, (msg) => {
     *   console.log('Importer event:', msg.event);
     * });
     * cleanup.pending.track('abc123');
     */
    listen(tract, optionsOrFn, fn) {
      const hasOptions = typeof optionsOrFn === 'object' && optionsOrFn !== null && typeof fn === 'function';
      const options = hasOptions ? optionsOrFn : {};
      const handler = hasOptions ? fn : optionsOrFn;

      if (typeof handler !== 'function') {
        throw new Error('appEvents.listen requires a callback');
      }

      const record = createListenerRecord(handler, {
        correlationKey: hasOptions ? options.correlationKey || 'use_id' : 'use_id',
        onDrop: hasOptions ? options.onDrop : null,
        onTimeout: hasOptions ? options.onTimeout : null,
        schema: hasOptions ? options.schema : null,
        timeout: hasOptions ? options.timeout : null
      });
      addListenerRecord(tract, record);

      const cleanup = () => {
        removeListenerRecord(tract, record);
      };
      cleanup.pending = record.pending;
      return cleanup;
    },

    unlisten(tract, fn) {
      if (!listeners[tract]) {
        return;
      }
      listeners[tract].slice().forEach(record => {
        if (record.fn === fn) {
          removeListenerRecord(tract, record);
        }
      });
    },

    onParseError(fn) {
      if (typeof fn !== 'function') {
        throw new Error('appEvents.onParseError requires a callback');
      }
      parseErrorHandlers.add(fn);
      return () => {
        parseErrorHandlers.delete(fn);
      };
    },

    onConnectionState(fn) {
      if (typeof fn !== 'function') {
        throw new Error('appEvents.onConnectionState requires a callback');
      }
      connectionStateHandlers.add(fn);
      fn({ connected: connectionState === 'connected', state: connectionState });
      return () => {
        connectionStateHandlers.delete(fn);
      };
    },

    getMetrics() {
      const now = Date.now();
      return {
        connected: connectionState === 'connected',
        state: connectionState,
        uptimeMs: connectedAt ? now - connectedAt : 0,
        lastMessageMs: lastMessageAt ? now - lastMessageAt : null,
        lastMessageAt: lastMessageAt,
        connectedAt: connectedAt
      };
    }
  };

  addListenerRecord('notification', createListenerRecord(function(msg) {
    window.AppServices?.notifications?.show(msg);
  }, {}));

  addListenerRecord('navigate', createListenerRecord(function(msg) {
    if (msg.facet && !msg.path) {
      window.selectFacet && window.selectFacet(msg.facet);
    } else if (msg.path) {
      if (msg.facet) {
        var expires = new Date();
        expires.setFullYear(expires.getFullYear() + 1);
        document.cookie = 'selectedFacet=' + msg.facet + '; expires=' + expires.toUTCString() + '; path=/; SameSite=Lax';
      }
      window.location.href = msg.path;
    }
  }, {}));

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connect);
  } else {
    connect();
  }
})();
