# 姿态分类结果平滑
class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes arre replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data

