class EMA:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.values = {}

    def update(self, metric_dict: dict) -> dict:
        for name, val in metric_dict.items():
            if name not in self.values:
                self.values[name] = val
            else:
                self.values[name] = self.alpha * val + (1 - self.alpha) * self.values[name]
        return self.values

    def get(self) -> dict:
        return self.values
