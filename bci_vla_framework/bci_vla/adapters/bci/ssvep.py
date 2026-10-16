"""SSVEP boundary shell. Acquisition, stimulus drawing and calibration are absent."""
from .replay import ReplayBCIAdapter


class SSVEPAdapterShell(ReplayBCIAdapter):
    """Accepts explicit protocol messages; preserves raw scores without decoding."""
