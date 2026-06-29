"""
backends/cuda_gst.py
====================
CUDA GStreamer backend: hardware decode on NVDEC (nvh26xdec) + GPU color
convert/scale (the cuda* family), then cudadownload copies the frame back to
system memory so it arrives as a numpy array. NVIDIA GPU required, but NO
DeepStream SDK -- this uses mainline GStreamer's nvcodec plugin.

All the appsink / threading / bus machinery lives in GstAppsinkCapture; this
file supplies only the decode + convert run of elements -- the ONE thing that
differs from the CPU backend.

Pipeline:
  <source front-end>
    ! nvh26xdec                                       (decode on the GPU)
    ! cudaconvert [! cudascale]                       (convert/resize on the GPU)
    ! video/x-raw(memory:CUDAMemory),format=<fmt>     (still in GPU memory)
    ! cudadownload                                    (GPU -> system memory copy)
    ! video/x-raw,format=<fmt>                        (now in CPU memory)
    ! appsink

Output is CPU numpy (cudadownload does the GPU->CPU copy). Zero-copy GPU / DLPack
output is the DeepStream backend's job, not this one.
"""

from ..config import Codec
from ._gst_appsink import GstAppsinkCapture

# Hardware NVDEC decoders (mainline nvcodec plugin), by codec.
_DECODER = {Codec.H264: "nvh264dec", Codec.H265: "nvh265dec"}


class CudaGstCapture(GstAppsinkCapture):

    def _media_chain(self, codec, fmt):
        # Decode on the GPU, then convert (and optionally scale) on the GPU while
        # the frame stays in CUDA memory, then download it to system memory.
        chain = [_DECODER[codec], "cudaconvert"]
        cuda_caps = f"video/x-raw(memory:CUDAMemory),format={fmt}"
        if self._config.width and self._config.height:
            chain.append("cudascale")
            cuda_caps += f",width={self._config.width},height={self._config.height}"
        chain.append(cuda_caps)
        chain.append("cudadownload")
        chain.append(f"video/x-raw,format={fmt}")        # now in system memory
        return chain