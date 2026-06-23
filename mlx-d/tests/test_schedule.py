"""Tests for the transfer schedule computation."""

import pytest

from mlx_d.utils import get_transfer_schedule


class TestTransferSchedule:
    """Tests for get_transfer_schedule — the linear noise schedule discretization."""

    def test_basic_even_division(self):
        """10 masks across 5 steps → 2 per step."""
        schedule = get_transfer_schedule(10, 5)
        assert schedule == [2, 2, 2, 2, 2]
        assert sum(schedule) == 10

    def test_remainder_front_loaded(self):
        """7 masks across 3 steps → [3, 2, 2] (remainder goes to earliest steps)."""
        schedule = get_transfer_schedule(7, 3)
        assert schedule == [3, 2, 2]
        assert sum(schedule) == 7

    def test_more_steps_than_masks(self):
        """3 masks across 10 steps → 3 steps with 1, 7 steps with 0."""
        schedule = get_transfer_schedule(3, 10)
        assert sum(schedule) == 3
        assert schedule.count(1) == 3
        assert schedule.count(0) == 7

    def test_one_step(self):
        """All masks committed in single step."""
        schedule = get_transfer_schedule(50, 1)
        assert schedule == [50]

    def test_zero_masks(self):
        """No masks → all zeros."""
        schedule = get_transfer_schedule(0, 5)
        assert schedule == [0, 0, 0, 0, 0]

    def test_zero_steps(self):
        """Zero steps → empty schedule."""
        schedule = get_transfer_schedule(10, 0)
        assert schedule == []

    def test_negative_inputs(self):
        """Negative inputs handled gracefully."""
        assert get_transfer_schedule(-5, 3) == [0, 0, 0]
        assert get_transfer_schedule(5, -3) == []

    def test_one_mask_many_steps(self):
        """Single mask across many steps."""
        schedule = get_transfer_schedule(1, 10)
        assert sum(schedule) == 1
        assert schedule[0] == 1  # front-loaded

    def test_large_values(self):
        """Large mask count and step count."""
        schedule = get_transfer_schedule(1000, 64)
        assert sum(schedule) == 1000
        assert len(schedule) == 64
        # All values should be either floor(1000/64)=15 or ceil=16
        assert all(v in (15, 16) for v in schedule)

    def test_schedule_is_monotonically_non_increasing(self):
        """Front-loaded remainder means schedule should be non-increasing."""
        for masks in range(1, 50):
            for steps in range(1, 20):
                schedule = get_transfer_schedule(masks, steps)
                for i in range(len(schedule) - 1):
                    assert schedule[i] >= schedule[i + 1], (
                        f"Non-monotonic at masks={masks}, steps={steps}: {schedule}"
                    )
