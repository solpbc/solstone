// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

(function() {
  const CHAT_REASON_DISPLAY_NAMES = Object.freeze({
    "google": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama"
  });

  const CHAT_REASONS = Object.freeze({
    "provider_key_invalid": {
      "template": "your {provider} key didn't validate",
      "action": {"label": "Open Settings", "href": "/app/settings/#providers"}
    },
    "provider_quota_exceeded": {
      "template": "your {provider} quota is spent — try again later",
      "action": null
    },
    "network_unreachable": {
      "template": "I couldn't reach the network",
      "action": null
    },
    "provider_response_invalid": {
      "template": "{provider} sent something I couldn't read — try again",
      "action": null
    },
    "provider_unavailable": {
      "template": "{provider} is having trouble — try again",
      "action": null
    },
    "chat_timeout": {
      "template": "chat took too long — try again",
      "action": null
    },
    "unknown": {
      "template": "chat had trouble — try again",
      "action": null
    }
  });

  window.CHAT_REASON_DISPLAY_NAMES = CHAT_REASON_DISPLAY_NAMES;
  window.CHAT_REASONS = CHAT_REASONS;
  window.renderChatReason = function(code, provider) {
    const reason = CHAT_REASONS[code];
    if (!reason) {
      return {code: code, message: code, action: null};
    }
    const providerSlug = String(provider || "");
    if (code === "unknown") {
      const displayName = CHAT_REASON_DISPLAY_NAMES[providerSlug];
      const message = displayName
        ? `something went wrong with ${displayName}`
        : reason.template;
      return {code: code, message: message, action: null};
    }
    const displayName = CHAT_REASON_DISPLAY_NAMES[providerSlug] || providerSlug;
    const message = reason.template.replace(/\{provider\}/g, displayName);
    const action = reason.action
      ? {label: reason.action.label, href: reason.action.href}
      : null;
    return {code: code, message: message, action: action};
  };
})();
