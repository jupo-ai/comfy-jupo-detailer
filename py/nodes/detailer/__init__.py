from . import cropper
from . import detailer
from . import outpainter
from . import stitcher

nodes = [
    cropper.DetailerCropper, 
    detailer.Detailer, 
    outpainter.DetailerOutpainter,
    stitcher.DetailerStitcher, 
]
