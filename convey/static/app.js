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

    // Find selected facet data
    const selectedFacetData = selectedFacet ? activeFacets.find(f => f.name === selectedFacet) : null;

    // Apply theme by updating CSS variables
    if (selectedFacetData && selectedFacetData.color) {
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
      pill.className = 'facet-pill' + (selectedFacet === facet.name ? ' selected' : '');

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

      // Apply color with opacity if facet is selected and has a color
      if (selectedFacet === facet.name && facet.color) {
        pill.style.background = hexToRgba(facet.color, 0.2);
        pill.style.borderColor = facet.color;
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

    // Find selected facet data
    const selectedFacetData = selectedFacet ? activeFacets.find(f => f.name === selectedFacet) : null;

    // Apply theme by updating CSS variables
    if (selectedFacetData && selectedFacetData.color) {
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

    // Update pill selection states
    pills.forEach((pill, index) => {
      const facetName = activeFacets[index]?.name;

      // Update selected class
      if (selectedFacet === facetName) {
        pill.classList.add('selected');

        // Apply color styling if selected and has color
        if (selectedFacetData && selectedFacetData.color) {
          pill.style.background = hexToRgba(selectedFacetData.color, 0.2);
          pill.style.borderColor = selectedFacetData.color;
        } else {
          pill.style.background = '';
          pill.style.borderColor = '';
        }
      } else {
        pill.classList.remove('selected');
        pill.style.background = '';
        pill.style.borderColor = '';
      }
    });
  }

  // Handle facet selection
  function selectFacet(facet) {
    selectedFacet = facet;
    saveSelectedFacetToCookie(facet);
    updateFacetSelection();
  }

  // Toggle sidebar
  function toggleSidebar() {
    document.body.classList.toggle('sidebar-open');
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

    // Hamburger menu interactions
    const hamburger = document.getElementById('hamburger');
    if (hamburger) {
      hamburger.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleSidebar();
      });
    }

    // App icon click - clear facet selection
    const appIcon = document.querySelector('.facet-bar .app-icon');
    if (appIcon) {
      appIcon.addEventListener('click', (e) => {
        selectedFacet = null;
        saveSelectedFacetToCookie(null);
        updateFacetSelection();
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

    // Close sidebar when clicking outside
    document.addEventListener('click', (e) => {
      if (document.body.classList.contains('sidebar-open')) {
        const menuBar = document.querySelector('.menu-bar');
        const facetBar = document.querySelector('.facet-bar');
        if (menuBar && facetBar && !menuBar.contains(e.target) && !facetBar.contains(e.target)) {
          document.body.classList.remove('sidebar-open');
        }
      }
    });
  }

  // Run initialization when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
