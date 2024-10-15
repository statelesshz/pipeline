import os
import re
from typing import Any, Optional

from loguru import logger
from openmind.utils.hub import OpenMindHub


TRUST_REMOTE_CODE = os.environ.get("OM_TRUST_REMOTE_CODE", "true").lower() in (
    "true",
    "1",
    "t",
    "on",
)


class Registry:
    """
    A utility class to register and retrieve values by keys.
    """

    def __init__(self):
        self._map = {}

    def register(self, keys: Any, value: Any):
        try:
            _ = iter(keys)
            is_iterable = True
        except TypeError:
            is_iterable = False

        if isinstance(keys, str) or not is_iterable:
            keys = [keys]

        for key in keys:
            if key in self._map:
                logger.warning(
                    f'Overriding previously registered "{key}" value "{value}"'
                )
            self._map[key] = value

    def get(self, key: Any) -> Any:
        if key in self._map:
            return self._map[key]
        return None

    def keys(self):
        return self._map.keys()


def download_from_repo(repo_id, revision=None, cache_dir=None, force_download=False):
    if not os.path.exists(repo_id):
        local_path = OpenMindHub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
        )
    else:
        local_path = repo_id
    return local_path


def get_task_from_readme(model_name) -> Optional[str]:
    """
    Get the task of the model by reading the README.md file.
    """
    task = None
    if not os.path.exists(model_name):
        task = OpenMindHub.get_task_from_repo(model_name)
    else:
        readme_file = os.path.join(model_name, "README.md")
        if os.path.exists(readme_file):
            with open(readme_file, "r") as file:
                content = file.read()
                pipeline_tag = re.search(r"pipeline_tag:\s?(([a-z]*-)*[a-z]*)", content)
                if pipeline_tag:
                    task = pipeline_tag.group(1)
    if task is None:
        logger.warning("Cannot infer the task from the provided model, please provide the task explicitly.")

    return task
