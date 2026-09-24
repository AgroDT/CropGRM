from typing import Optional
import logging
import ee
import geedim

logger = logging.getLogger(__name__)

class _BaseGEEExtractor:
    _is_initialized = False

    def __init__(
        self,
        project_id: Optional[str] = None,
        opt_url: str = "https://earthengine-highvolume.googleapis.com",
    ):
        self.project_id = project_id
        self.opt_url = opt_url
        self._initialize_ee()

    def _initialize_ee(self) -> None:
        if _BaseGEEExtractor._is_initialized:
            return
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id, opt_url=self.opt_url)
            else:
                ee.Initialize(opt_url=self.opt_url)
        except Exception:
            ee.Authenticate()
            if self.project_id:
                ee.Initialize(project=self.project_id, opt_url=self.opt_url)
            else:
                ee.Initialize(opt_url=self.opt_url)

        _BaseGEEExtractor._is_initialized = True
        logger.info("Google Earth Engine initialized.")