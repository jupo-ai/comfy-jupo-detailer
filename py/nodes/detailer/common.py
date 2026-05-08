from ...utils import mk_name, mk_category
from comfy_api.latest import io

PACKAGE_NAME = "Detailer"
CATEGORY = mk_category(PACKAGE_NAME)

IO_CROP_INFO = io.Custom("CROP_INFO")
