from .blueprint_core import BlueprintProcessor
from .blueprint_map import BlueprintMapMixin
from .blueprint_io import BlueprintIOMixin

class FullBlueprintProcessor(BlueprintProcessor, BlueprintMapMixin, BlueprintIOMixin):
    pass
