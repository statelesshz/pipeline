import importlib.metadata
import os
import operator
import re
from typing import Any, Optional
from packaging import version

from loguru import logger
from openmind.utils.hub import OpenMindHub


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

# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/versions.py#L49
ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None or want_ver is None:
        raise ValueError(
            f"Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This is unusual. Consider"
            f" reinstalling {pkg}."
        )
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )


def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the *site-packages* dir via *importlib.metadata*.

    Args:
        requirement (`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (`str`, *optional*): what suggestion to print in case of requirements not being met

    Example:

    ```python
    require_version("pandas>1.1.2")
    require_version("numpy>1.18.5", "this is important to have for whatever reason")
    ```"""

    hint = f"\n{hint}" if hint is not None else ""

    # non-versioned check
    if re.match(r"^[\w_\-\d]+$", requirement):
        pkg, op, want_ver = requirement, None, None
    else:
        match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2}.+)", requirement)
        if not match:
            raise ValueError(
                "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but"
                f" got {requirement}"
            )
        pkg, want_full = match[0]
        want_range = want_full.split(",")  # there could be multiple requirements
        wanted = {}
        for w in want_range:
            match = re.findall(r"^([\s!=<>]{1,2})(.+)", w)
            if not match:
                raise ValueError(
                    "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23,"
                    f" but got {requirement}"
                )
            op, want_ver = match[0]
            wanted[op] = want_ver
            if op not in ops:
                raise ValueError(f"{requirement}: need one of {list(ops.keys())}, but got {op}")

    # special case
    if pkg == "python":
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return

    # check if any version is installed
    try:
        got_ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        raise importlib.metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # check that the right version is installed if version number or a range was provided
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
