// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

/**
 * App System JavaScript
 * Handles facet selection, menu interactions, and responsive UI for app.html
 *
 * Requires:
 * - window.facetsData - Array of facet objects from server
 * - window.selectedFacet - Currently selected facet name or null (initialized by server)
 * - window.appFacetCounts - Object mapping facet names to counts (injected per-app)
 *
 * Public API:
 * - window.selectedFacet - Current facet selection (read/write)
 * - window.selectFacet(name) - Change facet selection programmatically
 * - 'facet.switch' event - Dispatched when selection changes
 */

(function(){
  // Facet filtering state
  let activeFacets = [];
  let previousFacet = null; // Track previous facet for toggle restoration

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
    const selectedFacetData = window.selectedFacet ? activeFacets.find(f => f.name === window.selectedFacet) : null;

    // Apply theme by updating CSS variables (only if facets are enabled)
    if (!facetsDisabled && selectedFacetData && selectedFacetData.color) {
      const color = selectedFacetData.color;
      const bgColor = color + '1a';  // 10% opacity

      document.documentElement.style.setProperty('--facet-color', color);
      document.documentElement.style.setProperty('--facet-bg', bgColor);
      document.documentElement.style.setProperty('--facet-border', color);
    } else {
      // Clear facet variables to use defaults
      // Remove server-rendered theme block so CSS defaults take over
      document.getElementById('facet-theme')?.remove();
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
      } else if (!window.selectedFacet) {
        // All-facet mode - all pills show as selected
        pill.className = 'facet-pill selected';
      } else {
        // Specific-facet mode - only selected pill highlights
        pill.className = 'facet-pill' + (window.selectedFacet === facet.name ? ' selected' : '');
      }

      // Add muted class if facet is muted
      if (facet.muted) {
        pill.classList.add('muted');
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

      const label = document.createElement('span');
      label.className = 'label';
      label.textContent = facet.title;
      pill.appendChild(label);

      // Apply color styling (only if facets enabled)
      if (!facetsDisabled && facet.color) {
        if (!window.selectedFacet) {
          // All-facet mode - each pill lights up with its own color
          pill.style.background = hexToRgba(facet.color, 0.2);
          pill.style.borderColor = facet.color;
          pill.style.boxShadow = `0 2px 6px ${hexToRgba(facet.color, 0.2)}`;
        } else if (window.selectedFacet === facet.name) {
          // Specific-facet mode - only selected pill colored
          pill.style.background = hexToRgba(facet.color, 0.2);
          pill.style.borderColor = facet.color;
        }
      }

      pill.onclick = () => selectFacet(facet.name);

      // Setup for drag-and-drop (attributes added here, listeners added in init)
      pill.dataset.facetName = facet.name;
      pill.draggable = true;

      facetPillsContainer.appendChild(pill);
    });

    // Add "+" button to create new facets (only if facets enabled)
    if (!facetsDisabled) {
      const addButton = document.createElement('div');
      addButton.className = 'facet-add-pill';
      addButton.textContent = '+';
      addButton.title = 'Create new facet';
      addButton.onclick = () => openFacetCreateModal();
      facetPillsContainer.appendChild(addButton);
    }
  }

  // Update selection styles without re-rendering
  function updateFacetSelection() {
    const container = document.querySelector('.facet-pills-container');
    if (!container) return;

    const pills = container.querySelectorAll('.facet-pill');

    // Check if facets are disabled for this app
    const facetsDisabled = document.querySelector('.facet-bar')?.classList.contains('facets-disabled');

    // Find selected facet data
    const selectedFacetData = window.selectedFacet ? activeFacets.find(f => f.name === window.selectedFacet) : null;

    // Update facet-bar class for all-facet mode
    const facetBar = document.querySelector('.facet-bar');
    if (facetBar && !facetsDisabled) {
      if (!window.selectedFacet) {
        facetBar.classList.add('all-facet-mode');
      } else {
        facetBar.classList.remove('all-facet-mode');
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
      // Remove server-rendered theme block so CSS defaults take over
      document.getElementById('facet-theme')?.remove();
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
      } else if (!window.selectedFacet) {
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
        if (window.selectedFacet === facetName) {
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

  // Update all-facet toggle visual state
  function updateAllFacetToggle(isHover = false) {
    const toggle = document.querySelector('.all-facet-toggle');
    if (!toggle) return;

    const facetsDisabled = document.querySelector('.facet-bar')?.classList.contains('facets-disabled');
    if (facetsDisabled) {
      toggle.style.display = 'none';
      return;
    }

    if (!window.selectedFacet) {
      // All-facet mode active
      toggle.textContent = isHover ? '⚪' : '🔘';
      toggle.setAttribute('data-active', 'true');
      toggle.title = previousFacet ? `Switch to ${activeFacets.find(f => f.name === previousFacet)?.title || previousFacet}` : 'Switch to first facet';
    } else {
      // Specific facet mode
      toggle.textContent = isHover ? '🔘' : '⚪';
      toggle.removeAttribute('data-active');
      toggle.title = 'Show all facets';
    }
  }

  // Handle facet selection
  function selectFacet(facet) {
    // Save previous facet before changing (only when switching to all-facet mode)
    if (facet === null && window.selectedFacet !== null) {
      previousFacet = window.selectedFacet;
    }

    window.selectedFacet = facet;
    saveSelectedFacetToCookie(facet);

    // Notify backend immediately (non-blocking, best-effort)
    fetch('/api/config/facets/select', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({facet: facet})
    }).catch(() => {}); // Ignore errors - cookie sync is fallback

    updateFacetSelection();
    updateAllFacetToggle();

    // Dispatch custom event for apps to listen to facet changes
    const facetData = facet ? activeFacets.find(f => f.name === facet) : null;
    window.dispatchEvent(new CustomEvent('facet.switch', {
      detail: {
        facet: facet,
        facetData: facetData
      }
    }));
  }

  // Generic drag-and-drop setup for reordering items
  function setupDragDrop(config) {
    const {
      container,           // Container element
      itemSelector,        // '.menu-item' or '.facet-pill'
      dataAttribute,       // 'appName' or 'facetName'
      onReorder,          // Callback with new order array
      preventDefault,     // Optional click prevention
      constrainDrop       // Optional constraint function(draggedItem, targetItem, items) -> constrainedTarget
    } = config;

    let draggedItem = null;
    let touchedItem = null;
    let touchDragActive = false;
    let isDragging = false;

    // Helper: Get current order and trigger callback
    function triggerReorder() {
      const items = Array.from(container.querySelectorAll(itemSelector));
      const order = items.map(item => item.dataset[dataAttribute]);
      onReorder(order);
    }

    // Prevent text selection during drag (but allow drag to start)
    container.addEventListener('selectstart', (e) => {
      const target = e.target.closest(itemSelector);
      if (target) {
        e.preventDefault(); // Prevent text selection
      }
    });

    // Click prevention (if needed)
    if (preventDefault) {
      container.addEventListener('click', (e) => {
        if (isDragging) {
          e.preventDefault();
          e.stopPropagation();
          isDragging = false;
        }
      }, true);
    }

    // Mouse drag-and-drop
    container.addEventListener('dragstart', (e) => {
      // For menu items, only allow drag from the drag handle
      if (itemSelector === '.menu-item') {
        const dragHandle = e.target.closest('.drag-handle');
        if (!dragHandle) {
          e.preventDefault();
          return;
        }

        // Only allow drag when menu is full
        if (!document.body.classList.contains('menu-full')) {
          e.preventDefault();
          return;
        }
      }

      const target = e.target.closest(itemSelector);
      if (!target) return;

      draggedItem = target;
      isDragging = true;

      target.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', '');

      // Create a better drag image
      const dragImage = target.cloneNode(true);
      dragImage.style.position = 'absolute';
      dragImage.style.top = '-9999px';
      dragImage.style.left = '-9999px';
      dragImage.style.opacity = '0.8';
      dragImage.style.transform = 'rotate(3deg)';
      dragImage.style.pointerEvents = 'none';
      document.body.appendChild(dragImage);

      const rect = target.getBoundingClientRect();
      e.dataTransfer.setDragImage(dragImage, rect.width / 2, rect.height / 2);

      // Remove the clone after drag image is captured
      setTimeout(() => dragImage.remove(), 0);
    });

    container.addEventListener('dragover', (e) => {
      e.preventDefault();
      let target = e.target.closest(itemSelector);
      if (!target || target === draggedItem) return;

      const items = Array.from(container.querySelectorAll(itemSelector));

      // Apply constraint if provided
      if (constrainDrop) {
        target = constrainDrop(draggedItem, target, items);
        if (!target || target === draggedItem) return;
      }

      // Remove drag-over from all items
      container.querySelectorAll(itemSelector).forEach(item => item.classList.remove('drag-over'));
      target.classList.add('drag-over');

      // Live reordering: move the dragged item in DOM as we drag over targets
      const draggedIndex = items.indexOf(draggedItem);
      const targetIndex = items.indexOf(target);

      if (draggedIndex !== -1 && targetIndex !== -1 && draggedIndex !== targetIndex) {
        if (draggedIndex < targetIndex) {
          // Moving down/right: insert after target
          container.insertBefore(draggedItem, target.nextSibling);
        } else {
          // Moving up/left: insert before target
          container.insertBefore(draggedItem, target);
        }
      }
    });

    container.addEventListener('drop', (e) => {
      e.preventDefault();

      // DOM already reordered during dragover, just trigger callback
      triggerReorder();
    });

    container.addEventListener('dragend', (e) => {
      const target = e.target.closest(itemSelector);
      if (!target) return;

      target.classList.remove('dragging');
      container.querySelectorAll(itemSelector).forEach(item => item.classList.remove('drag-over'));

      draggedItem = null;

      // Reset isDragging after a short delay to allow click prevention
      setTimeout(() => { isDragging = false; }, 100);
    });

    // Touch drag-and-drop
    container.addEventListener('touchstart', (e) => {
      // For menu items, only allow drag from the drag handle
      if (itemSelector === '.menu-item') {
        const dragHandle = e.target.closest('.drag-handle');
        if (!dragHandle) {
          return;
        }

        // Only allow drag when menu is full
        if (!document.body.classList.contains('menu-full')) {
          return;
        }
      }

      const target = e.target.closest(itemSelector);
      if (!target) return;

      touchedItem = target;
      touchDragActive = false;

      // Wait 200ms to distinguish tap from drag
      setTimeout(() => {
        if (touchedItem === target) {
          touchDragActive = true;
          isDragging = true;
          target.classList.add('dragging');
        }
      }, 200);
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
      if (!touchDragActive || !touchedItem) return;
      e.preventDefault();

      const touch = e.touches[0];
      const elementAtPoint = document.elementFromPoint(touch.clientX, touch.clientY);
      let target = elementAtPoint?.closest(itemSelector);

      if (target && target !== touchedItem) {
        const items = Array.from(container.querySelectorAll(itemSelector));

        // Apply constraint if provided
        if (constrainDrop) {
          target = constrainDrop(touchedItem, target, items);
          if (!target || target === touchedItem) return;
        }

        // Remove drag-over from all items
        container.querySelectorAll(itemSelector).forEach(item => item.classList.remove('drag-over'));
        target.classList.add('drag-over');

        // Live reordering during touch drag
        const draggedIndex = items.indexOf(touchedItem);
        const targetIndex = items.indexOf(target);

        if (draggedIndex !== -1 && targetIndex !== -1 && draggedIndex !== targetIndex) {
          if (draggedIndex < targetIndex) {
            container.insertBefore(touchedItem, target.nextSibling);
          } else {
            container.insertBefore(touchedItem, target);
          }
        }
      }
    }, { passive: false });

    container.addEventListener('touchend', (e) => {
      if (!touchDragActive || !touchedItem) {
        touchedItem = null;
        touchDragActive = false;
        return;
      }

      const touch = e.changedTouches[0];
      const elementAtPoint = document.elementFromPoint(touch.clientX, touch.clientY);
      const target = elementAtPoint?.closest(itemSelector);

      // DOM already reordered during touchmove, just trigger callback
      triggerReorder();

      // Cleanup
      touchedItem.classList.remove('dragging');
      container.querySelectorAll(itemSelector).forEach(item => item.classList.remove('drag-over'));
      touchedItem = null;
      touchDragActive = false;

      // Reset isDragging after a short delay
      setTimeout(() => { isDragging = false; }, 100);
    }, { passive: true });
  }

  // App starring state
  let starredApps = [];

  // Load starred apps from server-rendered data
  function loadStarredApps() {
    // Extract from menu items
    const menuItems = document.querySelectorAll('.menu-item[data-starred="true"]');
    starredApps = Array.from(menuItems).map(item => item.dataset.appName);
  }

  // Reorder menu items based on starred status
  function reorderMenuItems() {
    const menuBar = document.querySelector('.menu-bar');
    if (!menuBar) return;

    const menuItems = Array.from(menuBar.querySelectorAll('.menu-item'));

    // Separate starred and unstarred items
    const starredItems = menuItems.filter(item =>
      starredApps.includes(item.dataset.appName)
    );
    const unstarredItems = menuItems.filter(item =>
      !starredApps.includes(item.dataset.appName)
    );

    // Reorder: starred first, then unstarred
    const orderedItems = [...starredItems, ...unstarredItems];

    // Update DOM order
    orderedItems.forEach(item => {
      menuBar.appendChild(item);
    });

    // Update separator
    updateLastStarredSeparator();
  }

  // Toggle star status for an app
  async function toggleAppStar(appName) {
    const isStarred = starredApps.includes(appName);
    const newStarredStatus = !isStarred;

    // Optimistically update UI
    const menuItem = document.querySelector(`.menu-item[data-app-name="${appName}"]`);
    if (!menuItem) return;

    const starToggle = menuItem.querySelector('.star-toggle');
    if (!starToggle) return;

    // Update local state
    if (newStarredStatus) {
      starredApps.push(appName);
    } else {
      starredApps = starredApps.filter(name => name !== appName);
    }

    // Update DOM
    menuItem.dataset.starred = newStarredStatus;
    starToggle.textContent = newStarredStatus ? '★' : '☆';

    // Reorder menu items to reflect new grouping
    reorderMenuItems();

    // Save to backend
    try {
      const response = await fetch('/api/config/apps/star', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ app: appName, starred: newStarredStatus })
      });

      if (!response.ok) throw new Error('Failed to save star status');

      // No reload needed - DOM already updated

    } catch (error) {
      console.error('Failed to toggle app star:', error);
      if (window.AppServices?.notifications) {
        window.AppServices.notifications.show({
          app: 'system',
          title: 'Failed to save star status',
          message: error.message,
          autoDismiss: 5000
        });
      }

      // Revert optimistic update on error
      if (newStarredStatus) {
        starredApps = starredApps.filter(name => name !== appName);
      } else {
        starredApps.push(appName);
      }
      menuItem.dataset.starred = !newStarredStatus;
      starToggle.textContent = !newStarredStatus ? '★' : '☆';
      reorderMenuItems();
    }
  }

  // Update the last-starred class on menu items
  function updateLastStarredSeparator() {
    const menuItems = Array.from(document.querySelectorAll('.menu-item'));

    // Remove all last-starred classes
    menuItems.forEach(item => item.classList.remove('last-starred'));

    // Find last starred item
    let lastStarredIndex = -1;
    menuItems.forEach((item, index) => {
      if (item.dataset.starred === 'true') {
        lastStarredIndex = index;
      }
    });

    // Add class to last starred item
    if (lastStarredIndex >= 0 && starredApps.length > 0) {
      menuItems[lastStarredIndex].classList.add('last-starred');
    }
  }

  // Initialize
  function init() {
    // window.selectedFacet already initialized by server (see app.html)
    // Load facet chooser
    loadFacetChooser();

    // Load starred apps
    loadStarredApps();

    // Setup facet pill drag-and-drop
    const facetPillsContainer = document.querySelector('.facet-pills-container');
    if (facetPillsContainer) {
      setupDragDrop({
        container: facetPillsContainer,
        itemSelector: '.facet-pill',
        dataAttribute: 'facetName',
        preventDefault: true,  // Prevent facet selection on drag
        onReorder: async (order) => {
          // Update local array to match new order
          activeFacets.sort((a, b) => {
            return order.indexOf(a.name) - order.indexOf(b.name);
          });

          // Re-render pills (maintains selection state)
          renderFacetChooser();

          // Save to backend
          try {
            const response = await fetch('/api/config/facets/order', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ order })
            });

            if (!response.ok) throw new Error('Failed to save facet order');
          } catch (error) {
            console.error('Failed to save facet order:', error);
            if (window.AppServices?.notifications) {
              window.AppServices.notifications.show({
                app: 'system',
                title: 'Failed to save facet order',
                message: error.message,
                autoDismiss: 5000
              });
            }
          }
        }
      });
    }

    // Hamburger and menu-bar elements
    const hamburger = document.getElementById('hamburger');
    const menuBar = document.querySelector('.menu-bar');

    // Hamburger menu interactions
    if (hamburger && menuBar) {
      // Click to toggle menu
      hamburger.addEventListener('click', (e) => {
        e.stopPropagation();
        // If menu-all is active, remove it before toggling to menu-full
        if (document.body.classList.contains('menu-all')) {
          document.body.classList.remove('menu-all');
          const menuExpander = document.querySelector('.menu-expander');
          if (menuExpander) menuExpander.textContent = '⏷';
        }
        document.body.classList.toggle('menu-full');
      });

      // Close menu when clicking outside
      document.addEventListener('click', (e) => {
        if (document.body.classList.contains('menu-full')) {
          if (!menuBar.contains(e.target) && !hamburger.contains(e.target)) {
            document.body.classList.remove('menu-full');
          }
        }
        // Also close menu-all when clicking outside
        if (document.body.classList.contains('menu-all')) {
          const menuExpander = document.querySelector('.menu-expander');
          if (!menuBar.contains(e.target) && (!menuExpander || !menuExpander.contains(e.target))) {
            document.body.classList.remove('menu-all');
            if (menuExpander) menuExpander.textContent = '⏷';
          }
        }
      });

      // Star toggle click handlers
      menuBar.addEventListener('click', (e) => {
        const starToggle = e.target.closest('.star-toggle');
        if (starToggle) {
          e.preventDefault();
          e.stopPropagation();
          const appName = starToggle.dataset.appName;
          if (appName) {
            toggleAppStar(appName);
          }
        }
      });
    }

    // Menu expander click (toggle menu-all state)
    const menuExpander = document.querySelector('.menu-expander');
    if (menuExpander && menuBar) {
      menuExpander.addEventListener('click', (e) => {
        e.stopPropagation();
        document.body.classList.toggle('menu-all');

        // Update arrow direction
        if (document.body.classList.contains('menu-all')) {
          menuExpander.textContent = '⏶'; // Up arrow when menu-all
        } else {
          menuExpander.textContent = '⏷'; // Down arrow when menu-minimal
        }
      });
    }

    // App ordering via drag-and-drop
    const menuItemsContainer = document.querySelector('.menu-bar .menu-items');
    if (menuItemsContainer) {
      // Helper: Save app order to config with starred/unstarred grouping
      async function saveAppOrder(order) {
        try {
          // Separate into starred and unstarred groups
          const starredOrder = order.filter(name => starredApps.includes(name));
          const unstarredOrder = order.filter(name => !starredApps.includes(name));

          // Combine: starred first, then unstarred
          const finalOrder = [...starredOrder, ...unstarredOrder];

          const response = await fetch('/api/config/apps/order', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ order: finalOrder })
          });

          if (!response.ok) {
            throw new Error('Failed to save app order');
          }

          // No reload needed - DOM already updated during drag

        } catch (error) {
          console.error('Failed to save app order:', error);
          if (window.AppServices?.notifications) {
            window.AppServices.notifications.show({
              app: 'system',
              title: 'Failed to save app order',
              message: error.message,
              autoDismiss: 5000
            });
          }
        }
      }

      // Setup drag-and-drop using shared helper with boundary constraints
      setupDragDrop({
        container: menuItemsContainer,
        itemSelector: '.menu-item',
        dataAttribute: 'appName',
        preventDefault: true,
        onReorder: saveAppOrder,
        // Constraint function: prevent crossing starred/unstarred boundary
        constrainDrop: (draggedItem, targetItem, items) => {
          const draggedApp = draggedItem.dataset.appName;
          const targetApp = targetItem.dataset.appName;

          const draggedIsStarred = starredApps.includes(draggedApp);
          const targetIsStarred = starredApps.includes(targetApp);

          // Find boundary index (first unstarred item)
          const boundaryIndex = items.findIndex(item =>
            !starredApps.includes(item.dataset.appName)
          );

          // If no boundary (all starred or all unstarred), allow any drop
          if (boundaryIndex === -1 || boundaryIndex === 0) {
            return targetItem;
          }

          // Get indices
          const draggedIndex = items.indexOf(draggedItem);
          const targetIndex = items.indexOf(targetItem);

          // Prevent starred from going below boundary
          if (draggedIsStarred && targetIndex >= boundaryIndex) {
            // Clamp to last starred position
            return items[boundaryIndex - 1];
          }

          // Prevent unstarred from going above boundary
          if (!draggedIsStarred && targetIndex < boundaryIndex) {
            // Clamp to first unstarred position
            return items[boundaryIndex];
          }

          // Same group, allow drop
          return targetItem;
        }
      });
    }

    // All-facet toggle click - toggle between all-facet and specific facet
    const allFacetToggle = document.querySelector('.all-facet-toggle');
    if (allFacetToggle) {
      allFacetToggle.addEventListener('click', () => {
        if (window.selectedFacet === null) {
          // Currently in all-facet mode, switch to previous or first facet
          const targetFacet = previousFacet || (activeFacets.length > 0 ? activeFacets[0].name : null);
          if (targetFacet) {
            selectFacet(targetFacet);
          }
        } else {
          // Currently in specific facet mode, switch to all-facet mode
          selectFacet(null);
        }
      });

      // Hover effect - show opposite state
      allFacetToggle.addEventListener('mouseenter', () => {
        updateAllFacetToggle(true);
      });

      allFacetToggle.addEventListener('mouseleave', () => {
        updateAllFacetToggle(false);
      });
    }
  }

  // Expose selectFacet globally for notifications and other services
  window.selectFacet = selectFacet;

  // ========== FACET CREATION MODAL ==========

  // Create modal element (once)
  function ensureFacetCreateModal() {
    if (document.getElementById('facetCreateModal')) return;

    const modal = document.createElement('div');
    modal.id = 'facetCreateModal';
    modal.className = 'facet-create-modal';
    modal.innerHTML = `
      <div class="facet-create-content">
        <h3>Create New Facet</h3>
        <div class="facet-create-field">
          <label for="facetCreateTitle">Title</label>
          <input type="text" id="facetCreateTitle" placeholder="e.g., Work Projects" autofocus>
          <div class="facet-create-slug" id="facetCreateSlug"></div>
          <div class="facet-create-error" id="facetCreateError"></div>
        </div>
        <div class="facet-create-buttons">
          <button class="facet-create-cancel" id="facetCreateCancel">Cancel</button>
          <button class="facet-create-submit" id="facetCreateSubmit" disabled>Create</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    // Wire up events
    const titleInput = document.getElementById('facetCreateTitle');
    const slugDisplay = document.getElementById('facetCreateSlug');
    const submitBtn = document.getElementById('facetCreateSubmit');
    const cancelBtn = document.getElementById('facetCreateCancel');
    const errorDisplay = document.getElementById('facetCreateError');

    // Live slug generation as user types
    titleInput.addEventListener('input', () => {
      const title = titleInput.value.trim();
      const slug = titleToSlug(title);
      if (slug) {
        slugDisplay.textContent = slug;
        slugDisplay.classList.add('has-slug');
      } else {
        slugDisplay.textContent = '';
        slugDisplay.classList.remove('has-slug');
      }
      submitBtn.disabled = !slug;
      errorDisplay.classList.remove('visible');
    });

    // Enter to submit
    titleInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !submitBtn.disabled) {
        e.preventDefault();
        submitFacetCreate();
      } else if (e.key === 'Escape') {
        closeFacetCreateModal();
      }
    });

    // Cancel button
    cancelBtn.addEventListener('click', closeFacetCreateModal);

    // Submit button
    submitBtn.addEventListener('click', submitFacetCreate);

    // Click outside to close
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeFacetCreateModal();
      }
    });
  }

  // Convert title to slug (kebab-case)
  function titleToSlug(title) {
    if (!title) return '';
    return title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }

  // Open the modal
  function openFacetCreateModal() {
    ensureFacetCreateModal();
    const modal = document.getElementById('facetCreateModal');
    const titleInput = document.getElementById('facetCreateTitle');
    const slugDisplay = document.getElementById('facetCreateSlug');
    const submitBtn = document.getElementById('facetCreateSubmit');
    const errorDisplay = document.getElementById('facetCreateError');

    // Reset form
    titleInput.value = '';
    slugDisplay.textContent = '';
    slugDisplay.classList.remove('has-slug');
    submitBtn.disabled = true;
    errorDisplay.classList.remove('visible');

    modal.classList.add('visible');
    titleInput.focus();
  }

  // Close the modal
  function closeFacetCreateModal() {
    const modal = document.getElementById('facetCreateModal');
    if (modal) {
      modal.classList.remove('visible');
    }
  }

  // Submit facet creation
  async function submitFacetCreate() {
    const titleInput = document.getElementById('facetCreateTitle');
    const submitBtn = document.getElementById('facetCreateSubmit');
    const errorDisplay = document.getElementById('facetCreateError');

    const title = titleInput.value.trim();
    if (!title) return;

    submitBtn.disabled = true;
    submitBtn.textContent = 'Creating...';

    try {
      const response = await fetch('/app/settings/api/facet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create facet');
      }

      // Success - close modal, select new facet, navigate to settings
      closeFacetCreateModal();

      // Add new facet to local data
      const newFacet = {
        name: data.facet,
        title: data.config.title,
        color: data.config.color,
        emoji: data.config.emoji,
        muted: false,
        count: 0
      };
      activeFacets.push(newFacet);
      window.facetsData = activeFacets;

      // Re-render facet bar
      renderFacetChooser();

      // Select the new facet
      selectFacet(data.facet);

      // Navigate to settings app to customize
      window.location.href = '/app/settings';

    } catch (error) {
      errorDisplay.textContent = error.message;
      errorDisplay.classList.add('visible');
      submitBtn.disabled = false;
      submitBtn.textContent = 'Create';
    }
  }

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
   * Notification system
   */
  notifications: {
    _stack: [],
    _history: JSON.parse(localStorage.getItem('solstone:notification_history') || '[]'),
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
      this._addToHistory(notif);
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
     * Get notification history (most recent first)
     * @returns {Array} Array of notification objects
     */
    getHistory() {
      return [...this._history].reverse();
    },

    /**
     * Add notification to history and persist
     * @private
     */
    _addToHistory(notif) {
      // Store minimal data for history (exclude runtime fields)
      const historyEntry = {
        app: notif.app,
        icon: notif.icon,
        title: notif.title,
        message: notif.message,
        action: notif.action,
        facet: notif.facet,
        timestamp: notif.timestamp
      };

      this._history.push(historyEntry);

      // Cap at 10 items
      if (this._history.length > 10) {
        this._history = this._history.slice(-10);
      }

      // Persist to localStorage
      try {
        localStorage.setItem('solstone:notification_history', JSON.stringify(this._history));
      } catch (e) {
        // localStorage may be full or disabled
        console.warn('[Notifications] Failed to persist history:', e);
      }
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

        // Select facet if specified (for facet-aware navigation)
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
          ${n.dismissible ? `<button class="notification-close" onclick="event.preventDefault(); event.stopPropagation(); window.AppServices.notifications.dismiss(${n.id});">×</button>` : ''}
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
      if (seconds < 60) return 'now';
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) return `${minutes}m`;
      const hours = Math.floor(minutes / 60);
      if (hours < 24) return `${hours}h`;
      const days = Math.floor(hours / 24);
      return `${days}d`;
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
  },

  /**
   * Submenu system for app quick-links
   * Allows apps to define contextual links that appear on hover over menu icons
   */
  submenus: {
    _data: {},  // {appName: [items]}

    /**
     * Set entire submenu for an app (replaces existing)
     * @param {string} appName - Name of the app
     * @param {Array} items - Array of submenu items
     */
    set(appName, items) {
      this._data[appName] = items.map((item, index) => ({
        ...item,
        order: item.order !== undefined ? item.order : index
      }));
      this._render(appName);
    },

    /**
     * Add or update a single submenu item
     * @param {string} appName - Name of the app
     * @param {object} item - Item to add/update (must have id)
     */
    upsert(appName, item) {
      if (!this._data[appName]) {
        this._data[appName] = [];
      }

      const existing = this._data[appName].find(i => i.id === item.id);
      if (existing) {
        Object.assign(existing, item);
      } else {
        this._data[appName].push({
          ...item,
          order: item.order !== undefined ? item.order : this._data[appName].length
        });
      }
      this._render(appName);
    },

    /**
     * Remove a submenu item by id
     * @param {string} appName - Name of the app
     * @param {string} itemId - ID of item to remove
     */
    remove(appName, itemId) {
      if (!this._data[appName]) return;
      this._data[appName] = this._data[appName].filter(i => i.id !== itemId);
      this._render(appName);
    },

    /**
     * Clear all submenu items for an app
     * @param {string} appName - Name of the app
     */
    clear(appName) {
      delete this._data[appName];
      this._render(appName);
    },

    /**
     * Get submenu items for an app
     * @param {string} appName - Name of the app
     * @returns {Array} Array of submenu items
     */
    get(appName) {
      return this._data[appName] || [];
    },

    /**
     * Render submenu for an app
     * @private
     */
    _render(appName) {
      // Defer render if DOM not ready
      if (document.readyState === 'loading') {
        const self = this;
        document.addEventListener('DOMContentLoaded', function() {
          self._render(appName);
        });
        return;
      }

      const menuItem = document.querySelector(`.menu-item[data-app-name="${appName}"]`);
      if (!menuItem) return;

      // Remove existing submenu (could be in body or menu item)
      const existingId = `menu-submenu-${appName}`;
      const existing = document.getElementById(existingId);
      if (existing) {
        existing.remove();
      }

      // Get items for this app
      const items = this._data[appName];
      if (!items || items.length === 0) return;

      // Sort by order
      const sorted = [...items].sort((a, b) => (a.order || 0) - (b.order || 0));

      // Create submenu container - append to body to escape overflow:hidden
      const submenu = document.createElement('div');
      submenu.className = 'menu-submenu';
      submenu.id = existingId;
      submenu.dataset.appName = appName;

      // Create items
      sorted.forEach(item => {
        const link = document.createElement('a');
        link.className = 'menu-submenu-item';
        link.href = item.href || '#';

        if (item.facet) {
          link.dataset.facet = item.facet;
        }

        // Build inner HTML
        let html = '';
        if (item.icon) {
          html += `<span class="submenu-icon">${item.icon}</span>`;
        }
        html += `<span class="submenu-label">${window.AppServices._escapeHtml(item.label)}</span>`;
        if (item.badge) {
          html += `<span class="submenu-badge">${item.badge}</span>`;
        }

        link.innerHTML = html;

        // Click handler for facet selection
        if (item.facet) {
          link.addEventListener('click', (e) => {
            if (window.selectFacet) {
              window.selectFacet(item.facet);
            }
          });
        }

        submenu.appendChild(link);
      });

      // Append to body instead of menu item
      document.body.appendChild(submenu);

      // Position submenu on hover
      const positionSubmenu = () => {
        const rect = menuItem.getBoundingClientRect();
        submenu.style.position = 'fixed';
        submenu.style.top = rect.top + 'px';
        submenu.style.left = rect.right + 'px';
      };

      // Show/hide on hover
      menuItem.addEventListener('mouseenter', () => {
        // Only show when menu is not full (labels not visible)
        if (document.body.classList.contains('menu-full')) {
          return;
        }
        positionSubmenu();
        submenu.classList.add('visible');
      });

      menuItem.addEventListener('mouseleave', (e) => {
        // Check if moving to submenu
        const related = e.relatedTarget;
        if (related && submenu.contains(related)) {
          return;
        }
        submenu.classList.remove('visible');
      });

      submenu.addEventListener('mouseleave', (e) => {
        // Check if moving back to menu item
        const related = e.relatedTarget;
        if (related && menuItem.contains(related)) {
          return;
        }
        submenu.classList.remove('visible');
      });

      // Keep submenu visible while hovering it
      submenu.addEventListener('mouseenter', () => {
        submenu.classList.add('visible');
      });
    }
  },

  /**
   * Badge system for app icons and facet pills
   * Unified API with parallel app/facet namespaces
   */
  badges: {
    /**
     * App icon badges in the menu bar
     */
    app: {
      _data: {},  // {appName: count}

      /**
       * Set badge count for an app
       * @param {string} appName - Name of the app
       * @param {number} count - Badge count (0 or falsy to hide)
       */
      set(appName, count) {
        if (count && count > 0) {
          this._data[appName] = count;
        } else {
          delete this._data[appName];
        }
        this._render(appName);
      },

      /**
       * Clear badge for an app
       * @param {string} appName - Name of the app
       */
      clear(appName) {
        delete this._data[appName];
        this._render(appName);
      },

      /**
       * Get badge count for an app
       * @param {string} appName - Name of the app
       * @returns {number} Badge count (0 if not set)
       */
      get(appName) {
        return this._data[appName] || 0;
      },

      /**
       * Render badge for an app
       * @private
       */
      _render(appName) {
        // Defer render if DOM not ready
        if (document.readyState === 'loading') {
          const self = this;
          document.addEventListener('DOMContentLoaded', function() {
            self._render(appName);
          });
          return;
        }

        const menuItem = document.querySelector(`.menu-item[data-app-name="${appName}"]`);
        if (!menuItem) return;

        // Find the icon container
        const iconContainer = menuItem.querySelector('.icon');
        if (!iconContainer) return;

        // Remove existing badge
        const existing = iconContainer.querySelector('.menu-badge');
        if (existing) {
          existing.remove();
        }

        // Get count for this app
        const count = this._data[appName];
        if (!count || count <= 0) return;

        // Create badge element
        const badge = document.createElement('span');
        badge.className = 'menu-badge';
        badge.textContent = count;

        iconContainer.appendChild(badge);
      }
    },

    /**
     * Facet pill badges in the facet bar
     */
    facet: {
      _data: {},  // {facetName: count}

      /**
       * Set badge count for a facet
       * @param {string} facetName - Name of the facet
       * @param {number} count - Badge count (0 or falsy to hide)
       */
      set(facetName, count) {
        if (count && count > 0) {
          this._data[facetName] = count;
        } else {
          delete this._data[facetName];
        }
        this._render(facetName);
      },

      /**
       * Clear badge for a facet
       * @param {string} facetName - Name of the facet
       */
      clear(facetName) {
        delete this._data[facetName];
        this._render(facetName);
      },

      /**
       * Get badge count for a facet
       * @param {string} facetName - Name of the facet
       * @returns {number} Badge count (0 if not set)
       */
      get(facetName) {
        return this._data[facetName] || 0;
      },

      /**
       * Render badge for a facet
       * @private
       */
      _render(facetName) {
        // Defer render if DOM not ready
        if (document.readyState === 'loading') {
          const self = this;
          document.addEventListener('DOMContentLoaded', function() {
            self._render(facetName);
          });
          return;
        }

        const facetPill = document.querySelector(`.facet-pill[data-facet-name="${facetName}"]`);
        if (!facetPill) return;

        let badge = facetPill.querySelector('.facet-badge');
        const count = this._data[facetName];

        if (!count || count <= 0) {
          // Hide or remove badge
          if (badge) {
            badge.style.display = 'none';
          }
          return;
        }

        // Create badge if needed
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'facet-badge';
          const emojiContainer = facetPill.querySelector('.emoji-container');
          if (emojiContainer) {
            emojiContainer.appendChild(badge);
          }
        }

        badge.textContent = count;
        badge.style.display = '';
      }
    }
  }
};

/**
 * Privacy Blur
 * Blurs page content when window loses focus for privacy.
 * Unblurs on drag-enter to support file drop interactions.
 * Disabled on mobile devices where overlapping windows don't exist.
 */
(function() {
  // Detect mobile: no hover capability or touch device with narrow viewport
  const isMobile = window.matchMedia('(hover: none)').matches ||
                   (navigator.maxTouchPoints > 0 && window.innerWidth < 1024);

  if (isMobile) return;

  function onBlur() {
    document.body.classList.add('privacy-blur');
  }

  function onFocus() {
    document.body.classList.remove('privacy-blur');
  }

  window.addEventListener('blur', onBlur);
  window.addEventListener('focus', onFocus);

  // Handle drag-and-drop: unblur when dragging into window
  document.addEventListener('dragenter', onFocus);
  document.addEventListener('dragleave', (e) => {
    // relatedTarget is null when leaving the document entirely
    if (!e.relatedTarget && !document.hasFocus()) {
      onBlur();
    }
  });

  // Handle initial state (page may load while window is not focused)
  if (!document.hasFocus()) {
    document.body.classList.add('privacy-blur');
  }
})();
