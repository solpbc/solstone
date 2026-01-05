# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/utils.py functions."""

from observe.utils import (
    assign_monitor_positions,
    parse_screen_filename,
)


class TestAssignMonitorPositions:
    """Test monitor position assignment algorithm."""

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert assign_monitor_positions([]) == []

    def test_single_monitor(self):
        """Single monitor always gets 'center'."""
        monitors = [{"id": "DP-1", "box": [0, 0, 1920, 1080]}]
        result = assign_monitor_positions(monitors)

        assert len(result) == 1
        assert result[0]["position"] == "center"

    def test_two_side_by_side(self):
        """Two side-by-side monitors get 'left' and 'right'."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "left"
        assert positions["DP-2"] == "right"

    def test_two_stacked_vertically(self):
        """Two stacked monitors get 'top' and 'bottom'."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [0, 1080, 1920, 2160]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "top"
        assert positions["DP-2"] == "bottom"

    def test_three_in_a_row(self):
        """Three monitors in a row get 'left', 'center', 'right'."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
            {"id": "DP-3", "box": [3840, 0, 5760, 1080]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "left"
        assert positions["DP-2"] == "center"
        assert positions["DP-3"] == "right"

    def test_2x2_grid(self):
        """2x2 grid gets corner positions."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
            {"id": "DP-3", "box": [0, 1080, 1920, 2160]},
            {"id": "DP-4", "box": [1920, 1080, 3840, 2160]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "left-top"
        assert positions["DP-2"] == "right-top"
        assert positions["DP-3"] == "left-bottom"
        assert positions["DP-4"] == "right-bottom"

    def test_offset_dual_monitors(self):
        """Offset dual monitors (different sizes) get distinct positions."""
        # Larger monitor on right, smaller on left offset down
        # Monitors touch at x=1920 but don't overlap horizontally
        monitors = [
            {"id": "DP-1", "box": [0, 200, 1920, 1280]},  # 1920x1080, offset down
            {"id": "DP-2", "box": [1920, 0, 4480, 1440]},  # 2560x1440
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        # No horizontal overlap (touch at x=1920), so no vertical labels
        # DP-1 has monitor to its right → "left"
        # DP-2 has monitor to its left → "right"
        assert positions["DP-1"] == "left"
        assert positions["DP-2"] == "right"

    def test_no_position_collisions_side_by_side(self):
        """Verify no collisions in typical dual side-by-side setup."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
        ]
        result = assign_monitor_positions(monitors)

        positions = [m["position"] for m in result]
        # All positions should be unique
        assert len(positions) == len(set(positions))

    def test_no_position_collisions_2x2(self):
        """Verify no collisions in 2x2 grid."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
            {"id": "DP-3", "box": [0, 1080, 1920, 2160]},
            {"id": "DP-4", "box": [1920, 1080, 3840, 2160]},
        ]
        result = assign_monitor_positions(monitors)

        positions = [m["position"] for m in result]
        assert len(positions) == len(set(positions))

    def test_preserves_existing_fields(self):
        """Extra fields in monitor dicts are preserved."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080], "extra": "data"},
        ]
        result = assign_monitor_positions(monitors)

        assert result[0]["extra"] == "data"
        assert result[0]["position"] == "center"

    def test_ultrawide_center_with_offset_sides(self):
        """Ultrawide center with offset side monitors (no overlap) gets no vertical labels."""
        # Real-world config: ultrawide center, two 1080p on sides offset vertically
        monitors = [
            {"id": "DP-3", "box": [1920, 0, 5360, 1440]},  # 3440x1440 ultrawide
            {"id": "HDMI-4", "box": [5360, 219, 7280, 1299]},  # 1920x1080, right
            {"id": "HDMI-2", "box": [0, 231, 1920, 1311]},  # 1920x1080, left
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        # Side monitors touch (don't overlap) with center, so no vertical labels
        assert positions["DP-3"] == "center"
        assert positions["HDMI-4"] == "right"
        assert positions["HDMI-2"] == "left"


class TestParseScreenFilename:
    """Test screen filename parsing for per-monitor files.

    Files are now always in segment directories with simple names:
    position_connector_screen.webm (e.g., center_DP-3_screen.webm)
    """

    def test_standard_format(self):
        """Parse standard per-monitor filename."""
        position, connector = parse_screen_filename("center_DP-3_screen")
        assert position == "center"
        assert connector == "DP-3"

    def test_left_position(self):
        """Parse left position filename."""
        position, connector = parse_screen_filename("left_HDMI-1_screen")
        assert position == "left"
        assert connector == "HDMI-1"

    def test_compound_position(self):
        """Parse compound position like left-top."""
        position, connector = parse_screen_filename("left-top_DP-1_screen")
        assert position == "left-top"
        assert connector == "DP-1"

    def test_macos_numeric_display_id(self):
        """Parse macOS numeric display ID."""
        position, connector = parse_screen_filename("center_1_screen")
        assert position == "center"
        assert connector == "1"

    def test_simple_screen_filename(self):
        """Simple screen filename without position returns unknown."""
        position, connector = parse_screen_filename("screen")
        assert position == "unknown"
        assert connector == "unknown"

    def test_audio_filename(self):
        """Audio filename returns unknown."""
        position, connector = parse_screen_filename("audio")
        assert position == "unknown"
        assert connector == "unknown"

    def test_right_position(self):
        """Parse right position filename."""
        position, connector = parse_screen_filename("right_HDMI-2_screen")
        assert position == "right"
        assert connector == "HDMI-2"

    def test_compound_left_bottom(self):
        """Parse compound left-bottom position."""
        position, connector = parse_screen_filename("left-bottom_DP-2_screen")
        assert position == "left-bottom"
        assert connector == "DP-2"
