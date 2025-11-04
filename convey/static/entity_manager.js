/**
 * Entity Manager - Drag and drop interface for managing entity aliases
 * Uses Pointer Events API for cross-device support (mouse + touch)
 */

// Entity type colors
const TYPE_COLORS = {
  Person: "#3b82f6", // blue
  Company: "#10b981", // green
  Project: "#8b5cf6", // purple
  Tool: "#f59e0b", // amber
  Location: "#ef4444", // red
  default: "#6b7280", // gray
};

// Drag state
let dragState = {
  isDragging: false,
  draggedElement: null,
  draggedEntityName: null,
  ghost: null,
  currentDropTarget: null,
  pointerId: null,
  startX: 0,
  startY: 0,
  offsetX: 0,
  offsetY: 0,
};

/**
 * Initialize the entity manager when DOM is ready
 */
document.addEventListener("DOMContentLoaded", function () {
  applyEntityColors();
  initializeDragAndDrop();
  initializeClickToEdit();
  initializeModals();
});

/**
 * Apply entity type colors to card borders
 */
function applyEntityColors() {
  const cards = document.querySelectorAll(".entity-card");

  cards.forEach((card) => {
    const type = card.dataset.entityType;
    const color = TYPE_COLORS[type] || TYPE_COLORS["default"];
    card.style.borderColor = color;
  });
}

/**
 * Set up drag and drop functionality using Pointer Events
 */
function initializeDragAndDrop() {
  const detectedCards = document.querySelectorAll(".detected");

  detectedCards.forEach((card) => {
    card.addEventListener("pointerdown", handlePointerDown);
  });

  // Global pointer move and up handlers
  document.addEventListener("pointermove", handlePointerMove);
  document.addEventListener("pointerup", handlePointerUp);
  document.addEventListener("pointercancel", handlePointerCancel);
}

/**
 * Handle pointer down (start of potential drag)
 */
function handlePointerDown(e) {
  // Only handle primary button/touch
  if (e.button !== 0 && e.pointerType === "mouse") return;

  const card = e.currentTarget;

  // Store initial state
  dragState.draggedElement = card;
  dragState.draggedEntityName = card.dataset.entityName;
  dragState.startX = e.clientX;
  dragState.startY = e.clientY;
  dragState.pointerId = e.pointerId;

  // Calculate offset from pointer to card's top-left
  const rect = card.getBoundingClientRect();
  dragState.offsetX = e.clientX - rect.left;
  dragState.offsetY = e.clientY - rect.top;

  // Capture pointer events to this element
  card.setPointerCapture(e.pointerId);

  // Prevent default to avoid text selection
  e.preventDefault();
}

/**
 * Handle pointer move (dragging)
 */
function handlePointerMove(e) {
  if (!dragState.draggedElement) return;

  const deltaX = Math.abs(e.clientX - dragState.startX);
  const deltaY = Math.abs(e.clientY - dragState.startY);

  // Start dragging if moved more than threshold (5px)
  if (!dragState.isDragging && (deltaX > 5 || deltaY > 5)) {
    startDragging();
  }

  if (dragState.isDragging) {
    // Update ghost position
    updateGhostPosition(e.clientX, e.clientY);

    // Find element under pointer
    const elementBelow = getElementBelowPointer(e.clientX, e.clientY);
    const dropTarget = elementBelow?.closest(".entity-card.attached");

    // Update drop target highlighting
    if (dropTarget !== dragState.currentDropTarget) {
      if (dragState.currentDropTarget) {
        dragState.currentDropTarget.classList.remove("drag-over");
      }
      if (dropTarget) {
        dropTarget.classList.add("drag-over");
      }
      dragState.currentDropTarget = dropTarget;
    }
  }
}

/**
 * Handle pointer up (end drag)
 */
function handlePointerUp(e) {
  if (!dragState.draggedElement) return;

  if (dragState.isDragging) {
    // Check if we're over a valid drop target
    if (dragState.currentDropTarget) {
      const targetName = dragState.currentDropTarget.dataset.entityName;
      const sourceName = dragState.draggedEntityName;

      if (sourceName && targetName && sourceName !== targetName) {
        showMergeModal(sourceName, targetName);
      }
    }

    endDragging();
  }

  resetDragState();
}

/**
 * Handle pointer cancel (drag interrupted)
 */
function handlePointerCancel(e) {
  if (dragState.isDragging) {
    endDragging();
  }
  resetDragState();
}

/**
 * Start the dragging operation
 */
function startDragging() {
  dragState.isDragging = true;
  dragState.draggedElement.classList.add("dragging");

  // Create ghost element
  const ghost = dragState.draggedElement.cloneNode(true);
  ghost.classList.add("drag-ghost");
  ghost.style.position = "fixed";
  ghost.style.pointerEvents = "none";
  ghost.style.zIndex = "10000";
  ghost.style.width = dragState.draggedElement.offsetWidth + "px";
  ghost.style.opacity = "0.8";
  ghost.style.transform = "scale(1.05)";

  document.body.appendChild(ghost);
  dragState.ghost = ghost;
}

/**
 * Update ghost element position
 */
function updateGhostPosition(clientX, clientY) {
  if (!dragState.ghost) return;

  const x = clientX - dragState.offsetX;
  const y = clientY - dragState.offsetY;

  dragState.ghost.style.left = x + "px";
  dragState.ghost.style.top = y + "px";
}

/**
 * Get the element below the pointer (excluding ghost)
 */
function getElementBelowPointer(clientX, clientY) {
  // Hide ghost temporarily to get element below
  if (dragState.ghost) {
    dragState.ghost.style.display = "none";
  }

  const element = document.elementFromPoint(clientX, clientY);

  // Restore ghost
  if (dragState.ghost) {
    dragState.ghost.style.display = "";
  }

  return element;
}

/**
 * End the dragging operation
 */
function endDragging() {
  // Remove ghost
  if (dragState.ghost) {
    dragState.ghost.remove();
    dragState.ghost = null;
  }

  // Remove drag-over state from all attached cards
  document.querySelectorAll(".attached").forEach((card) => {
    card.classList.remove("drag-over");
  });

  // Remove dragging state from dragged element
  if (dragState.draggedElement) {
    dragState.draggedElement.classList.remove("dragging");
  }
}

/**
 * Reset drag state
 */
function resetDragState() {
  // Release pointer capture if still active
  if (dragState.draggedElement && dragState.pointerId !== null) {
    try {
      dragState.draggedElement.releasePointerCapture(dragState.pointerId);
    } catch (e) {
      // Ignore - pointer may already be released
    }
  }

  dragState.isDragging = false;
  dragState.draggedElement = null;
  dragState.draggedEntityName = null;
  dragState.currentDropTarget = null;
  dragState.pointerId = null;
  dragState.startX = 0;
  dragState.startY = 0;
  dragState.offsetX = 0;
  dragState.offsetY = 0;
}

/**
 * Initialize click-to-edit functionality for attached entities
 */
function initializeClickToEdit() {
  const attachedCards = document.querySelectorAll(".attached");

  attachedCards.forEach((card) => {
    card.addEventListener("click", function (e) {
      // Don't trigger if clicking during drag
      if (card.classList.contains("dragging")) {
        return;
      }

      const entityName = card.dataset.entityName;
      const currentAka = card.dataset.aka || "";

      showAkaEditModal(entityName, currentAka);
    });
  });
}

/**
 * Initialize modal event listeners
 */
function initializeModals() {
  // Merge modal
  const mergeModal = document.getElementById("merge-modal");
  const confirmMerge = document.getElementById("confirm-merge");
  const cancelMerge = document.getElementById("cancel-merge");

  if (cancelMerge) {
    cancelMerge.addEventListener("click", closeMergeModal);
  }

  if (confirmMerge) {
    confirmMerge.addEventListener("click", function () {
      const sourceName = document.getElementById("source-name").textContent;
      const targetName = document.getElementById("target-name").textContent;
      addAka(targetName, sourceName);
    });
  }

  // Click outside modal to close
  if (mergeModal) {
    mergeModal.addEventListener("click", function (e) {
      if (e.target === mergeModal) {
        closeMergeModal();
      }
    });
  }

  // AKA edit modal
  const akaModal = document.getElementById("aka-edit-modal");
  const akaInput = document.getElementById("aka-input");

  if (akaModal) {
    akaModal.addEventListener("click", function (e) {
      if (e.target === akaModal) {
        closeAkaEditModal();
      }
    });
  }

  if (akaInput) {
    akaInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        const entityName = document.getElementById("edit-entity-name").textContent;
        const akaList = akaInput.value.trim();
        updateAka(entityName, akaList);
      } else if (e.key === "Escape") {
        e.preventDefault();
        closeAkaEditModal();
      }
    });
  }
}

/**
 * Show the merge confirmation modal
 */
function showMergeModal(sourceName, targetName) {
  const modal = document.getElementById("merge-modal");
  const sourceEl = document.getElementById("source-name");
  const targetEl = document.getElementById("target-name");

  if (sourceEl) sourceEl.textContent = sourceName;
  if (targetEl) targetEl.textContent = targetName;

  modal.classList.remove("hidden");

  // Focus the confirm button
  setTimeout(() => {
    document.getElementById("confirm-merge").focus();
  }, 100);
}

/**
 * Close the merge confirmation modal
 */
function closeMergeModal() {
  const modal = document.getElementById("merge-modal");
  modal.classList.add("hidden");
}

/**
 * Show the AKA edit modal
 */
function showAkaEditModal(entityName, currentAka) {
  const modal = document.getElementById("aka-edit-modal");
  const nameEl = document.getElementById("edit-entity-name");
  const input = document.getElementById("aka-input");

  if (nameEl) nameEl.textContent = entityName;
  if (input) {
    input.value = currentAka;
    modal.classList.remove("hidden");
    // Focus and select all text
    setTimeout(() => {
      input.focus();
      input.select();
    }, 100);
  }
}

/**
 * Close the AKA edit modal
 */
function closeAkaEditModal() {
  const modal = document.getElementById("aka-edit-modal");
  modal.classList.add("hidden");
}

/**
 * Add a detected entity name to an attached entity's aka list
 */
function addAka(targetEntity, sourceEntity) {
  const confirmBtn = document.getElementById("confirm-merge");
  const originalText = confirmBtn.textContent;

  confirmBtn.textContent = "Adding...";
  confirmBtn.disabled = true;

  fetch(`/api/facets/${encodeURIComponent(facetName)}/entities/manage/add-aka`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      target_entity: targetEntity,
      source_entity: sourceEntity,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        // Remove the detected entity card from DOM with animation
        const sourceCard = document.querySelector(
          `.detected[data-entity-name="${CSS.escape(sourceEntity)}"]`
        );
        if (sourceCard) {
          sourceCard.style.opacity = "0";
          sourceCard.style.transform = "scale(0.8)";
          setTimeout(() => {
            sourceCard.remove();

            // Check if detected grid is now empty
            const detectedGrid = document.getElementById("detected-grid");
            const remainingDetected = detectedGrid.querySelectorAll(".detected");
            if (remainingDetected.length === 0) {
              detectedGrid.innerHTML = `
                <div class="empty-state">
                  <p>No detected entities in the last 30 days.</p>
                </div>
              `;
            }
          }, 300);
        }

        // Update the attached card's aka count
        updateAkaCount(targetEntity);

        closeMergeModal();
      } else {
        alert("Error: " + (data.error || "Unknown error"));
        confirmBtn.textContent = originalText;
        confirmBtn.disabled = false;
      }
    })
    .catch((error) => {
      console.error("Error adding aka:", error);
      alert("Network error: " + error.message);
      confirmBtn.textContent = originalText;
      confirmBtn.disabled = false;
    });
}

/**
 * Update an attached entity's aka list directly
 */
function updateAka(entityName, akaList) {
  const modal = document.getElementById("aka-edit-modal");

  fetch(`/api/facets/${encodeURIComponent(facetName)}/entities/manage/update-aka`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      entity_name: entityName,
      aka_list: akaList,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        // Update the card's data-aka attribute
        const card = document.querySelector(
          `.attached[data-entity-name="${CSS.escape(entityName)}"]`
        );
        if (card) {
          const akaArray = data.aka || [];
          card.dataset.aka = akaArray.join(",");

          // Update aka count display
          updateAkaCount(entityName);
        }

        closeAkaEditModal();
      } else {
        alert("Error: " + (data.error || "Unknown error"));
      }
    })
    .catch((error) => {
      console.error("Error updating aka:", error);
      alert("Network error: " + error.message);
    });
}

/**
 * Update the aka count display on an attached entity card
 */
function updateAkaCount(entityName) {
  const card = document.querySelector(
    `.attached[data-entity-name="${CSS.escape(entityName)}"]`
  );
  if (!card) return;

  const akaString = card.dataset.aka || "";
  const akaArray = akaString ? akaString.split(",") : [];
  const count = akaArray.filter((a) => a.trim()).length;

  // Remove existing aka count if present
  const existingCount = card.querySelector(".entity-aka-count");
  if (existingCount) {
    existingCount.remove();
  }

  // Add new count if > 0
  if (count > 0) {
    const countEl = document.createElement("div");
    countEl.className = "entity-aka-count";
    countEl.textContent = `(${count} aka)`;
    card.appendChild(countEl);
  }
}
