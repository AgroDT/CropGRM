import logging
from typing import Optional

import ee

logger = logging.getLogger(__name__)


class _BaseGEEExtractor:
    """Share Earth Engine initialization across data extractors."""

    _initialized_config = None

    def __init__(
        self,
        project_id: Optional[str] = None,
        opt_url: str = "https://earthengine-highvolume.googleapis.com",
    ):
        """Configure and initialize the Earth Engine connection.

        Args:
            project_id (str | None): Earth Engine project ID; None uses the
                initialization default.
            opt_url (str): Earth Engine API endpoint URL.

        Raises:
            ValueError: If this process already initialized a different project or URL.
        """
        self.project_id = project_id
        self.opt_url = opt_url
        self._initialize_ee()

    def _initialize_ee(self) -> None:
        """Reuse the process-global session only for the same project and endpoint."""
        config = (self.project_id, self.opt_url)
        initialized = _BaseGEEExtractor._initialized_config
        if initialized is not None:
            if initialized != config:
                raise ValueError(
                    "Earth Engine is already initialized with "
                    f"project={initialized[0]!r}, endpoint={initialized[1]!r}; "
                    f"requested project={config[0]!r}, endpoint={config[1]!r}. "
                    "Use a separate process for a different Earth Engine session."
                )
            return

        ee.Initialize(project=self.project_id, opt_url=self.opt_url)

        _BaseGEEExtractor._initialized_config = config
        logger.info("Google Earth Engine initialized.")
