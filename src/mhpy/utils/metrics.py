class EMA:
    def __init__(self, alpha: float = 0.1):
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"
        self.alpha = alpha
        self.beta = 1.0 - self.alpha
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metric_dict: dict[str, float]) -> dict[str, float]:
        corrected_values = {}

        for name, val in metric_dict.items():
            self.counts[name] = self.counts.get(name, 0) + 1

            prev_val = self.values.get(name, 0.0)
            new_val = self.alpha * val + self.beta * prev_val
            self.values[name] = new_val

            correction_factor = 1 - (self.beta ** self.counts[name])

            if correction_factor > 1e-8:
                corrected_values[name] = new_val / correction_factor
            else:
                corrected_values[name] = new_val

        return corrected_values
