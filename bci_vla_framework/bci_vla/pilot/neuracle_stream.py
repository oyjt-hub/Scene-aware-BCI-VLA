"""Bounded receive-only Neuracle stream inspection; no triggers or commands."""
import math
import struct
from ..protocols import require,integer


class FloatFrameParser:
    def __init__(self,channels):
        integer(channels,1);require(channels<=256,'unsupported_channel_count')
        self.channels=channels;self.tail=b'';self.frames=0;self.nonfinite=0

    def feed(self,data):
        require(isinstance(data,bytes),'bytes_required')
        joined=self.tail+data;stride=4*self.channels;length=len(joined)//stride*stride
        body,self.tail=joined[:length],joined[length:]
        rows=list(struct.iter_unpack('<'+'f'*self.channels,body))
        self.frames+=len(rows);self.nonfinite+=sum(not math.isfinite(v) for row in rows for v in row)
        return rows
