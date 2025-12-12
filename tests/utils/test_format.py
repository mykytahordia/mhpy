from mhpy.utils.format import fcount
from mhpy.utils.format import fsize


class TestFcount:
    def test_fcount_zero(self):
        assert fcount(0) == "0"

    def test_fcount_small_numbers(self):
        assert fcount(1) == "1"
        assert fcount(10) == "10"
        assert fcount(100) == "100"
        assert fcount(999) == "999"

    def test_fcount_thousands(self):
        assert fcount(1_000) == "1k"
        assert fcount(1_500) == "1.5k"
        assert fcount(10_000) == "10k"
        assert fcount(100_000) == "100k"
        assert fcount(999_999) == "999.999k"

    def test_fcount_millions(self):
        assert fcount(1_000_000) == "1M"
        assert fcount(1_500_000) == "1.5M"
        assert fcount(10_000_000) == "10M"
        assert fcount(100_000_000) == "100M"

    def test_fcount_billions(self):
        assert fcount(1_000_000_000) == "1B"
        assert fcount(1_500_000_000) == "1.5B"
        assert fcount(10_000_000_000) == "10B"

    def test_fcount_trillions(self):
        assert fcount(1_000_000_000_000) == "1T"
        assert fcount(1_500_000_000_000) == "1.5T"

    def test_fcount_petabytes(self):
        assert fcount(1_000_000_000_000_000) == "1P"

    def test_fcount_exceeds_units(self):
        result = fcount(1_000_000_000_000_000_000)
        assert result == "1E"

    def test_fcount_negative_numbers(self):
        assert fcount(-1) == "-1"
        assert fcount(-1000) == "-1k"
        assert fcount(-1_000_000) == "-1M"


class TestFsize:
    def test_fsize_zero(self):
        assert fsize(0) == "0B"

    def test_fsize_bytes(self):
        assert fsize(1) == "1B"
        assert fsize(100) == "100B"
        assert fsize(1023) == "1023B"

    def test_fsize_kilobytes(self):
        assert fsize(1024) == "1KB"
        assert fsize(1536) == "1.5KB"
        assert fsize(10240) == "10KB"

    def test_fsize_megabytes(self):
        assert fsize(1024 * 1024) == "1MB"
        assert fsize(int(1.5 * 1024 * 1024)) == "1.5MB"
        assert fsize(10 * 1024 * 1024) == "10MB"

    def test_fsize_gigabytes(self):
        assert fsize(1024**3) == "1GB"
        assert fsize(int(1.5 * 1024**3)) == "1.5GB"
        assert fsize(10 * 1024**3) == "10GB"

    def test_fsize_terabytes(self):
        assert fsize(1024**4) == "1TB"
        assert fsize(int(1.5 * 1024**4)) == "1.5TB"

    def test_fsize_petabytes(self):
        assert fsize(1024**5) == "1PB"

    def test_fsize_exabytes(self):
        assert fsize(1024**6) == "1EB"

    def test_fsize_zettabytes(self):
        assert fsize(1024**7) == "1ZB"

    def test_fsize_yottabytes(self):
        assert fsize(1024**8) == "1YB"

    def test_fsize_exceeds_units(self):
        # When size exceeds all units, it should clamp to last unit (YB)
        result = fsize(1024**9)
        assert "YB" in result
