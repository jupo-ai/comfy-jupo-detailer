from comfy_api.latest import ComfyExtension

from .py import NODES


class Extension(ComfyExtension):
    async def get_node_list(self):
        return NODES
    
async def comfy_entrypoint():
    return Extension()


WEB_DIRECTORY = "./web"
