"""
backends/cpu_gst.py
===================
CPU GStreamer backend: software decode (avdec) + CPU color convert
(videoconvert), delivering numpy frames. No NVIDIA dependency -- runs anywhere.

All the appsink / threading / bus machinery lives in GstAppsinkCapture; this
file supplies only the decode + convert run of elements.

Pipeline:
  <source front-end> ! avdec_h26x ! videoconvert [! videoscale] ! <caps> ! appsink
"""

from ..config import Codec
from ._gst_appsink import GstAppsinkCapture

# CPU software decoders, by codec.
_DECODER = {Codec.H264: "avdec_h264", Codec.H265: "avdec_h265"}


class CpuGstCapture(GstAppsinkCapture):

    def _media_chain(self, codec, fmt):
        # avdec decodes to raw video (CPU), videoconvert produces the requested
        # packed format; videoscale is only added when a resize was requested.
        chain = [_DECODER[codec], "videoconvert"]
        caps = f"video/x-raw,format={fmt}"
        if self._config.width and self._config.height:
            chain.append("videoscale")
            caps += f",width={self._config.width},height={self._config.height}"
        chain.append(caps)
        return chain