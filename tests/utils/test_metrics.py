import pytest

from mhpy.utils.metrics import EMA


class TestEMA:
    def test_ema_initialization_default(self):
        ema = EMA()
        assert ema.alpha == 0.1
        assert ema.beta == 0.9
        assert ema.values == {}
        assert ema.counts == {}

    def test_ema_initialization_custom_alpha(self):
        alpha = 0.5
        ema = EMA(alpha=alpha)
        assert ema.alpha == alpha
        assert ema.beta == 0.5
        assert ema.values == {}
        assert ema.counts == {}

    def test_ema_update_first_value(self):
        ema = EMA(alpha=0.1)
        metric_dict = {"loss": 1.0, "accuracy": 0.9}

        result = ema.update(metric_dict)

        assert result["loss"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(0.9)
        assert ema.values["loss"] == pytest.approx(0.1)
        assert ema.values["accuracy"] == pytest.approx(0.09)

    def test_ema_update_subsequent_values(self):
        ema = EMA(alpha=0.1)

        ema.update({"loss": 1.0})
        result = ema.update({"loss": 2.0})

        raw_val = 0.1 * 2.0 + 0.9 * 0.1
        correction_factor = 1 - 0.9**2
        assert result["loss"] == pytest.approx(raw_val / correction_factor)
        assert ema.values["loss"] == pytest.approx(raw_val)

    def test_ema_update_multiple_metrics(self):
        ema = EMA(alpha=0.2)

        ema.update({"loss": 1.0, "accuracy": 0.8, "f1": 0.75})
        result = ema.update({"loss": 0.5, "accuracy": 0.9, "f1": 0.85})

        loss_raw = 0.2 * 0.5 + 0.8 * 0.2
        acc_raw = 0.2 * 0.9 + 0.8 * 0.16
        f1_raw = 0.2 * 0.85 + 0.8 * 0.15
        correction = 1 - 0.8**2

        assert result["loss"] == pytest.approx(loss_raw / correction)
        assert result["accuracy"] == pytest.approx(acc_raw / correction)
        assert result["f1"] == pytest.approx(f1_raw / correction)

    def test_ema_update_new_metric_added(self):
        ema = EMA(alpha=0.1)

        ema.update({"loss": 1.0})
        result = ema.update({"loss": 2.0, "accuracy": 0.9})

        assert "accuracy" in result
        assert result["accuracy"] == pytest.approx(0.9)

        loss_raw = 0.1 * 2.0 + 0.9 * 0.1
        assert result["loss"] == pytest.approx(loss_raw / (1 - 0.9**2))

    def test_ema_alpha_zero(self):
        ema = EMA(alpha=0.0)

        ema.update({"loss": 1.0})
        result = ema.update({"loss": 2.0})

        raw_val = 0.0 * 2.0 + 1.0 * 0.0
        correction = 1 - 1.0**2
        assert result["loss"] == pytest.approx(raw_val / correction if correction != 0 else 0.0)

    def test_ema_alpha_one(self):
        ema = EMA(alpha=1.0)

        ema.update({"loss": 1.0})
        result = ema.update({"loss": 2.0})

        raw_val = 1.0 * 2.0 + 0.0 * 1.0
        correction = 1 - 0.0**2
        assert result["loss"] == pytest.approx(raw_val / correction)

    def test_ema_sequence_of_updates(self):
        ema = EMA(alpha=0.5)

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            ema.update({"metric": val})

        expected_raw = 4.03125
        assert ema.values["metric"] == pytest.approx(expected_raw)

    def test_ema_empty_update(self):
        ema = EMA(alpha=0.1)

        result = ema.update({})

        assert result == {}
        assert ema.values == {}
