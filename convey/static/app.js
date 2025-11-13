/**
 * App System JavaScript
 * Handles facet selection, menu interactions, and responsive UI for app.html
 *
 * Requires:
 * - window.facetsData - Array of facet objects from server
 * - window.selectedFacetFromServer - Currently selected facet name or null
 * - window.appFacetCounts - Object mapping facet names to counts (injected per-app)
 */

(function(){
  // Facet filtering state
  let activeFacets = [];
  let selectedFacet = null; // null means "All"

  // Save facet selection to cookie (server-driven)
  function saveSelectedFacetToCookie(facet) {
    if (facet) {
      const expires = new Date();
      expires.setFullYear(expires.getFullYear() + 1);
      document.cookie = `selectedFacet=${facet}; expires=${expires.toUTCString()}; path=/; SameSite=Lax`;
    } else {
      document.cookie = 'selectedFacet=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Lax';
    }
  }

  // Convert hex color to rgba with opacity
  function hexToRgba(hex, alpha) {
    if (!hex || hex.length < 6) return `rgba(128,128,128,${alpha})`;
    const r = parseInt(hex.substring(1,3), 16);
    const g = parseInt(hex.substring(3,5), 16);
    const b = parseInt(hex.substring(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // Load facets from embedded data
  function loadFacetChooser() {
    activeFacets = window.facetsData || [];

    // Enrich facets with app-specific counts (injected by app.html)
    const appCounts = window.appFacetCounts || {};
    activeFacets.forEach(facet => {
      facet.count = appCounts[facet.name] || 0;
    });

    renderFacetChooser();
  }

  // Render facet pills in top bar
  function renderFacetChooser() {
    const facetPillsContainer = document.querySelector('.facet-pills-container');
    if (!facetPillsContainer) return;

    facetPillsContainer.innerHTML = '';

    // Check if facets are disabled for this app
    const facetsDisabled = document.querySelector('.facet-bar')?.classList.contains('facets-disabled');

    // Find selected facet data
    const selectedFacetData = selectedFacet ? activeFacets.find(f => f.name === selectedFacet) : null;

    // Apply theme by updating CSS variables (only if facets are enabled)
    if (!facetsDisabled && selectedFacetData && selectedFacetData.color) {
      const color = selectedFacetData.color;
      const bgColor = color + '1a';  // 10% opacity

      document.documentElement.style.setProperty('--facet-color', color);
      document.documentElement.style.setProperty('--facet-bg', bgColor);
      document.documentElement.style.setProperty('--facet-border', color);
    } else {
      // Clear facet variables to use defaults
      document.documentElement.style.removeProperty('--facet-color');
      document.documentElement.style.removeProperty('--facet-bg');
      document.documentElement.style.removeProperty('--facet-border');
    }

    // Facet pills
    activeFacets.forEach(facet => {
      const pill = document.createElement('div');
      // When disabled, no pill gets 'selected' class - all look identical
      if (facetsDisabled) {
        pill.className = 'facet-pill';
      } else if (!selectedFacet) {
        // All-facet mode - all pills show as selected
        pill.className = 'facet-pill selected';
      } else {
        // Specific-facet mode - only selected pill highlights
        pill.className = 'facet-pill' + (selectedFacet === facet.name ? ' selected' : '');
      }

      if (facet.emoji) {
        const emojiContainer = document.createElement('div');
        emojiContainer.className = 'emoji-container';

        const emoji = document.createElement('span');
        emoji.className = 'emoji';
        emoji.textContent = facet.emoji;
        emojiContainer.appendChild(emoji);

        // Add badge if count > 0
        const count = facet.count || 0;
        if (count > 0) {
          const badge = document.createElement('span');
          badge.className = 'facet-badge';
          badge.textContent = count;
          emojiContainer.appendChild(badge);
        }

        pill.appendChild(emojiContainer);
      }

      const title = document.createElement('span');
      title.textContent = facet.title;
      pill.appendChild(title);

      // Apply color styling (only if facets enabled)
      if (!facetsDisabled && facet.color) {
        if (!selectedFacet) {
          // All-facet mode - each pill lights up with its own color
          pill.style.background = hexToRgba(facet.color, 0.2);
          pill.style.borderColor = facet.color;
          pill.style.boxShadow = `0 2px 6px ${hexToRgba(facet.color, 0.2)}`;
        } else if (selectedFacet === facet.name) {
          // Specific-facet mode - only selected pill colored
          pill.style.background = hexToRgba(facet.color, 0.2);
          pill.style.borderColor = facet.color;
        }
      }

      pill.onclick = () => selectFacet(facet.name);
      facetPillsContainer.appendChild(pill);
    });
  }

  // Update selection styles without re-rendering
  function updateFacetSelection() {
    const container = document.querySelector('.facet-pills-container');
    if (!container) return;

    const pills = container.querySelectorAll('.facet-pill');

    // Check if facets are disabled for this app
    const facetsDisabled = document.querySelector('.facet-bar')?.classList.contains('facets-disabled');

    // Find selected facet data
    const selectedFacetData = selectedFacet ? activeFacets.find(f => f.name === selectedFacet) : null;

    // Update facet-bar class for all-facet mode
    const facetBar = document.querySelector('.facet-bar');
    if (facetBar && !facetsDisabled) {
      if (!selectedFacet) {
        facetBar.classList.add('all-facet-mode');
      } else {
        facetBar.classList.remove('all-facet-mode');
      }
    }

    // Update app-icon tooltip
    const appIcon = document.querySelector('.facet-bar .app-icon');
    if (appIcon && !facetsDisabled) {
      if (selectedFacet) {
        appIcon.setAttribute('title', 'Show all facets');
      } else {
        appIcon.removeAttribute('title');
      }
    }

    // Apply theme by updating CSS variables (only if facets are enabled)
    if (!facetsDisabled && selectedFacetData && selectedFacetData.color) {
      const color = selectedFacetData.color;
      const bgColor = color + '1a';  // 10% opacity

      document.documentElement.style.setProperty('--facet-color', color);
      document.documentElement.style.setProperty('--facet-bg', bgColor);
      document.documentElement.style.setProperty('--facet-border', color);
    } else {
      // Clear facet variables to use defaults (all-facet mode or no color)
      document.documentElement.style.removeProperty('--facet-color');
      document.documentElement.style.removeProperty('--facet-bg');
      document.documentElement.style.removeProperty('--facet-border');
    }

    // Update pill selection states (only if facets enabled)
    pills.forEach((pill, index) => {
      const facet = activeFacets[index];
      const facetName = facet?.name;

      if (facetsDisabled) {
        // When disabled, ensure no pill has 'selected' class
        pill.classList.remove('selected');
        pill.style.background = '';
        pill.style.borderColor = '';
        pill.style.boxShadow = '';
      } else if (!selectedFacet) {
        // All-facet mode - all pills light up with their own colors
        pill.classList.add('selected');

        if (facet && facet.color) {
          pill.style.background = hexToRgba(facet.color, 0.2);
          pill.style.borderColor = facet.color;
          pill.style.boxShadow = `0 2px 6px ${hexToRgba(facet.color, 0.2)}`;
        } else {
          pill.style.background = '';
          pill.style.borderColor = '';
          pill.style.boxShadow = '';
        }
      } else {
        // Specific-facet mode - only selected pill highlights
        if (selectedFacet === facetName) {
          pill.classList.add('selected');

          // Apply color styling if selected and has color
          if (selectedFacetData && selectedFacetData.color) {
            pill.style.background = hexToRgba(selectedFacetData.color, 0.2);
            pill.style.borderColor = selectedFacetData.color;
            pill.style.boxShadow = '';
          } else {
            pill.style.background = '';
            pill.style.borderColor = '';
            pill.style.boxShadow = '';
          }
        } else {
          pill.classList.remove('selected');
          pill.style.background = '';
          pill.style.borderColor = '';
          pill.style.boxShadow = '';
        }
      }
    });
  }

  // Handle facet selection
  function selectFacet(facet) {
    selectedFacet = facet;
    saveSelectedFacetToCookie(facet);
    updateFacetSelection();

    // Dispatch custom event for apps to listen to facet changes
    const facetData = facet ? activeFacets.find(f => f.name === facet) : null;
    window.dispatchEvent(new CustomEvent('facet.switch', {
      detail: {
        facet: facet,
        facetData: facetData
      }
    }));
  }

  // Collapse facet pills when container is too narrow
  function collapseFacetPills() {
    const container = document.querySelector('.facet-pills-container');
    if (!container) return;

    const pills = Array.from(container.querySelectorAll('.facet-pill'));
    if (pills.length === 0) return;

    // Reset all pills to full display
    pills.forEach(pill => pill.classList.remove('icon-only'));

    // Force a reflow to get accurate measurements
    container.offsetWidth;

    // Check if we're overflowing
    const containerWidth = container.clientWidth;
    let totalWidth = 0;

    pills.forEach(pill => {
      totalWidth += pill.offsetWidth + 8; // Include margin
    });

    // If overflowing, collapse pills from right to left
    if (totalWidth > containerWidth) {
      // Start from the end (right side) and collapse until we fit
      for (let i = pills.length - 1; i >= 0; i--) {
        const pill = pills[i];

        pill.classList.add('icon-only');

        // Force reflow and recalculate
        container.offsetWidth;

        totalWidth = 0;
        pills.forEach(p => {
          totalWidth += p.offsetWidth + 8;
        });

        // If we fit now, stop collapsing
        if (totalWidth <= containerWidth) break;
      }
    }
  }

  // Initialize
  function init() {
    // Initialize facet selection from server
    selectedFacet = window.selectedFacetFromServer;

    // Load facet chooser
    loadFacetChooser();

    // Set up ResizeObserver to collapse pills when container width changes
    const facetPillsContainer = document.querySelector('.facet-pills-container');
    if (facetPillsContainer) {
      const resizeObserver = new ResizeObserver(() => {
        collapseFacetPills();
      });
      resizeObserver.observe(facetPillsContainer);
    }

    // Initial collapse check after DOM settles
    setTimeout(collapseFacetPills, 0);

    // Menu state tracking
    let isClickMode = false;
    let hideTimeout = null;

    // Hamburger and menu-bar elements
    const hamburger = document.getElementById('hamburger');
    const menuBar = document.querySelector('.menu-bar');

    // Hamburger menu interactions
    if (hamburger && menuBar) {
      // Hover to show menu (hover mode)
      hamburger.addEventListener('mouseenter', () => {
        if (!isClickMode) {
          clearTimeout(hideTimeout);
          document.body.classList.add('sidebar-open');
        }
      });

      // Auto-dismiss when leaving hamburger (hover mode)
      hamburger.addEventListener('mouseleave', () => {
        if (!isClickMode) {
          hideTimeout = setTimeout(() => {
            if (!isClickMode) {
              document.body.classList.remove('sidebar-open');
            }
          }, 200);
        }
      });

      // Cancel auto-dismiss when entering menu (hover mode)
      menuBar.addEventListener('mouseenter', () => {
        if (!isClickMode) {
          clearTimeout(hideTimeout);
        }
      });

      // Auto-dismiss when leaving menu (hover mode)
      menuBar.addEventListener('mouseleave', () => {
        if (!isClickMode) {
          hideTimeout = setTimeout(() => {
            if (!isClickMode) {
              document.body.classList.remove('sidebar-open');
            }
          }, 200);
        }
      });

      // Click to toggle and lock menu (click mode)
      hamburger.addEventListener('click', (e) => {
        e.stopPropagation();
        clearTimeout(hideTimeout);

        if (isClickMode) {
          // Already locked, clicking closes it
          document.body.classList.remove('sidebar-open');
          isClickMode = false;
        } else {
          // Lock menu open
          document.body.classList.add('sidebar-open');
          isClickMode = true;
        }
      });

      // Close sidebar when clicking outside
      document.addEventListener('click', (e) => {
        if (document.body.classList.contains('sidebar-open')) {
          if (!menuBar.contains(e.target) && !hamburger.contains(e.target)) {
            document.body.classList.remove('sidebar-open');
            isClickMode = false;
            clearTimeout(hideTimeout);
          }
        }
      });
    }

    // App icon click - switch to all-facet mode
    const appIcon = document.querySelector('.facet-bar .app-icon');
    if (appIcon) {
      appIcon.addEventListener('click', (e) => {
        selectedFacet = null;
        saveSelectedFacetToCookie(null);
        updateFacetSelection();

        // Dispatch facet.switch event for all-facet mode
        window.dispatchEvent(new CustomEvent('facet.switch', {
          detail: {
            facet: null,
            facetData: null
          }
        }));
      });
    }

    // Handle submenu items with data-facet attribute
    document.querySelectorAll('.submenu-item[data-facet]').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const facetName = item.getAttribute('data-facet');
        const targetPath = item.getAttribute('href');

        // Select the facet (sets cookie and updates UI)
        selectFacet(facetName);

        // Navigate to the path
        window.location.href = targetPath;
      });
    });
  }

  // Expose selectFacet globally for notifications and other services
  window.selectFacet = selectFacet;

  // Run initialization when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/**
 * App Services Framework
 * Global API for apps to register background services, update badges, and show notifications
 */
window.AppServices = {
  services: {},

  /**
   * Register an app background service
   * @param {string} appName - Name of the app
   * @param {object} service - Service object with initialize() method
   */
  register(appName, service) {
    this.services[appName] = service;
    if (service.initialize) {
      try {
        service.initialize();
      } catch (err) {
        console.error(`[AppServices] Failed to initialize ${appName} service:`, err);
      }
    }
  },

  /**
   * Update badge count for a facet or app
   * @param {string} appName - Name of the app
   * @param {string|null} facetName - Facet name, or null for app-level badge
   * @param {number} count - Badge count (0 to hide)
   */
  updateBadge(appName, facetName, count) {
    if (facetName) {
      // Update facet pill badge
      const facetPill = document.querySelector(`.facet-pill[data-facet="${facetName}"]`);
      if (facetPill) {
        let badge = facetPill.querySelector('.facet-badge');
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'facet-badge';
          const emojiContainer = facetPill.querySelector('.emoji-container');
          if (emojiContainer) {
            emojiContainer.appendChild(badge);
          }
        }
        badge.textContent = count || '';
        badge.style.display = count > 0 ? 'inline-block' : 'none';
      }

      // Update submenu badge for facet
      const submenuItem = document.querySelector(
        `.menu-item[data-app="${appName}"] .submenu-item[data-facet="${facetName}"]`
      );
      if (submenuItem) {
        let badge = submenuItem.querySelector('.submenu-badge');
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'submenu-badge';
          submenuItem.appendChild(badge);
        }
        badge.textContent = count || '';
        badge.style.display = count > 0 ? 'inline-block' : 'none';
      }
    } else {
      // Update app-level badge in menu
      const menuItem = document.querySelector(`.menu-item[data-app="${appName}"]`);
      if (menuItem) {
        let badge = menuItem.querySelector('.app-badge');
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'app-badge';
          const link = menuItem.querySelector('a') || menuItem;
          link.appendChild(badge);
        }
        badge.textContent = count || '';
        badge.style.display = count > 0 ? 'inline-block' : 'none';
      }
    }
  },

  /**
   * Update submenu items for an app
   * @param {string} appName - Name of the app
   * @param {Array} items - Array of {label, path, facet?, count?} objects
   */
  updateSubmenu(appName, items) {
    const submenu = document.querySelector(`.menu-item[data-app="${appName}"] .submenu`);
    if (!submenu) return;

    submenu.innerHTML = items.map(item => `
      <div class="submenu-item" ${item.facet ? `data-facet="${item.facet}"` : ''}>
        <a href="${item.path}">${this._escapeHtml(item.label)}</a>
        ${item.count ? `<span class="submenu-badge">${item.count}</span>` : ''}
      </div>
    `).join('');
  },

  /**
   * Notification system
   */
  notifications: {
    _stack: [],
    _nextId: 1,
    _container: null,

    /**
     * Show a persistent notification card
     * @param {object} options - {app, icon, title, message, action, facet, dismissible, badge, autoDismiss}
     * @returns {number} Notification ID
     */
    show(options) {
      const notif = {
        id: this._nextId++,
        app: options.app || 'system',
        icon: options.icon || '📬',
        title: options.title || 'Notification',
        message: options.message || '',
        action: options.action || null,
        facet: options.facet || null,
        dismissible: options.dismissible !== false,
        badge: options.badge || null,
        timestamp: Date.now(),
        autoDismiss: options.autoDismiss || null
      };

      this._stack.push(notif);
      this._render();

      // Browser notification if permitted
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(notif.title, {
          body: notif.message,
          icon: notif.icon,
          tag: `${notif.app}-${notif.id}`
        });
      }

      // Auto-dismiss timer
      if (notif.autoDismiss) {
        setTimeout(() => this.dismiss(notif.id), notif.autoDismiss);
      }

      return notif.id;
    },

    /**
     * Dismiss a specific notification
     * @param {number} id - Notification ID
     */
    dismiss(id) {
      this._stack = this._stack.filter(n => n.id !== id);
      this._render();
    },

    /**
     * Dismiss all notifications for an app
     * @param {string} appName - App name
     */
    dismissApp(appName) {
      this._stack = this._stack.filter(n => n.app !== appName);
      this._render();
    },

    /**
     * Dismiss all notifications
     */
    dismissAll() {
      this._stack = [];
      this._render();
    },

    /**
     * Get count of active notifications
     * @returns {number}
     */
    count() {
      return this._stack.length;
    },

    /**
     * Update existing notification
     * @param {number} id - Notification ID
     * @param {object} options - Fields to update
     */
    update(id, options) {
      const notif = this._stack.find(n => n.id === id);
      if (!notif) return;

      Object.assign(notif, options);
      this._render();
    },

    /**
     * Render notification cards
     * @private
     */
    _render() {
      if (!this._container) {
        this._container = document.getElementById('notification-center');
        if (!this._container) return;
      }

      // Limit to 5 most recent
      const visible = this._stack.slice(-5);
      const visibleIds = visible.map(n => n.id);

      // Get existing card IDs
      const existingCards = Array.from(this._container.querySelectorAll('.notification-card'));
      const existingIds = existingCards.map(card => parseInt(card.getAttribute('data-id')));

      // Remove cards that are no longer in visible stack
      existingCards.forEach(card => {
        const id = parseInt(card.getAttribute('data-id'));
        if (!visibleIds.includes(id)) {
          card.remove();
        }
      });

      // Add or update cards
      visible.forEach(n => {
        let card = this._container.querySelector(`.notification-card[data-id="${n.id}"]`);

        if (!card) {
          // New card - create and animate
          card = this._createCard(n);
          this._container.appendChild(card);
        } else {
          // Existing card - update content (no animation)
          this._updateCard(card, n);
        }
      });

      // Start timestamp updater if not already running
      if (visible.length > 0 && !this._updateInterval) {
        this._updateInterval = setInterval(() => this._updateTimestamps(), 60000);
      } else if (visible.length === 0 && this._updateInterval) {
        clearInterval(this._updateInterval);
        this._updateInterval = null;
      }
    },

    /**
     * Attach click handler to notification card
     * @private
     */
    _attachClickHandler(card, n) {
      if (!n.action) return;

      card.onclick = (e) => {
        // Ignore clicks on close button
        if (e.target.closest('.notification-close')) {
          return;
        }

        // Prevent default for anchor tags
        if (card.tagName === 'A') {
          e.preventDefault();
        }

        // Select facet if specified (like submenu navigation)
        if (n.facet && window.selectFacet) {
          window.selectFacet(n.facet);
        }

        // Navigate to the path
        window.location.href = n.action;
      };
    },

    /**
     * Create a new notification card element
     * @private
     */
    _createCard(n) {
      // Use anchor tag for semantic HTML when action exists
      const card = document.createElement(n.action ? 'a' : 'div');
      card.className = 'notification-card';
      card.setAttribute('data-id', n.id);
      card.setAttribute('data-app', n.app);

      if (n.action) {
        card.href = n.action;
        card.style.cursor = 'pointer';
        if (n.facet) {
          card.setAttribute('data-facet', n.facet);
        }
      }

      const relativeTime = this._getRelativeTime(n.timestamp);
      card.innerHTML = `
        <div class="notification-header">
          <span class="notification-app-icon">${n.icon}</span>
          <span class="notification-app-name">${window.AppServices._escapeHtml(n.app)}</span>
          ${n.dismissible ? `<button class="notification-close" onclick="window.AppServices.notifications.dismiss(${n.id}); event.stopPropagation();">×</button>` : ''}
        </div>
        <div class="notification-body">
          <div class="notification-title">${window.AppServices._escapeHtml(n.title)}</div>
          ${n.message ? `<div class="notification-message">${window.AppServices._escapeHtml(n.message)}</div>` : ''}
          ${n.badge ? `<span class="notification-badge">${n.badge}</span>` : ''}
        </div>
        <div class="notification-footer">
          <span class="notification-time">${relativeTime}</span>
        </div>
      `;

      // Attach click handler
      this._attachClickHandler(card, n);

      return card;
    },

    /**
     * Update existing notification card content
     * @private
     */
    _updateCard(card, n) {
      // Update title
      const titleEl = card.querySelector('.notification-title');
      if (titleEl) {
        titleEl.textContent = n.title;
      }

      // Update message
      const messageEl = card.querySelector('.notification-message');
      if (n.message) {
        if (messageEl) {
          messageEl.textContent = n.message;
        } else {
          const bodyEl = card.querySelector('.notification-body');
          const newMessage = document.createElement('div');
          newMessage.className = 'notification-message';
          newMessage.textContent = n.message;
          bodyEl.insertBefore(newMessage, bodyEl.querySelector('.notification-badge'));
        }
      } else if (messageEl) {
        messageEl.remove();
      }

      // Update badge
      const badgeEl = card.querySelector('.notification-badge');
      if (n.badge) {
        if (badgeEl) {
          badgeEl.textContent = n.badge;
        } else {
          const bodyEl = card.querySelector('.notification-body');
          const newBadge = document.createElement('span');
          newBadge.className = 'notification-badge';
          newBadge.textContent = n.badge;
          bodyEl.appendChild(newBadge);
        }
      } else if (badgeEl) {
        badgeEl.remove();
      }

      // Update time
      const timeEl = card.querySelector('.notification-time');
      if (timeEl) {
        timeEl.textContent = this._getRelativeTime(n.timestamp);
      }

      // Update action and facet
      if (n.action) {
        if (card.tagName === 'A') {
          card.href = n.action;
        }
        card.style.cursor = 'pointer';

        if (n.facet) {
          card.setAttribute('data-facet', n.facet);
        } else {
          card.removeAttribute('data-facet');
        }

        // Recreate click handler with new action/facet values
        this._attachClickHandler(card, n);
      } else {
        card.style.cursor = 'default';
        card.onclick = null;
      }
    },

    /**
     * Update timestamps on visible notifications
     * @private
     */
    _updateTimestamps() {
      const cards = this._container?.querySelectorAll('.notification-card');
      if (!cards) return;

      cards.forEach(card => {
        const id = parseInt(card.getAttribute('data-id'));
        const notif = this._stack.find(n => n.id === id);
        if (notif) {
          const timeEl = card.querySelector('.notification-time');
          if (timeEl) {
            timeEl.textContent = this._getRelativeTime(notif.timestamp);
          }
        }
      });
    },

    /**
     * Get relative time string
     * @private
     */
    _getRelativeTime(timestamp) {
      const seconds = Math.floor((Date.now() - timestamp) / 1000);
      if (seconds < 60) return 'just now';
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) return `${minutes}m ago`;
      const hours = Math.floor(minutes / 60);
      if (hours < 24) return `${hours}h ago`;
      const days = Math.floor(hours / 24);
      return `${days}d ago`;
    }
  },

  /**
   * Request browser notification permission
   * @returns {Promise<string>} Permission state
   */
  async requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
      return await Notification.requestPermission();
    }
    return Notification.permission;
  },

  /**
   * Escape HTML to prevent XSS
   * @private
   */
  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
};
