"""General utility functions """
import pathlib


def create_dir(full_dir_path: str,
               make_new: bool = True,
               try_limit: int = 1000) -> str:
    """Create a directory.

    If the directory already exists and make_new is False, then does nothing.
    Else if the directory already exists and make_new is True then creates a
    new directory by adding '_{i}' where i is lowest integer that does not
    yet already exist.

    Returns the full directory path of the created directory
    """
    try:
        path = pathlib.Path(full_dir_path)
        path.mkdir(parents=True, exist_ok=False)
        return full_dir_path
    except FileExistsError:
        if not make_new:
            return full_dir_path

    count = 0
    while count < try_limit:
        new_full_dir_path = f"{full_dir_path}_{count}"
        try:
            path = pathlib.Path(new_full_dir_path)
            path.mkdir(parents=True, exist_ok=False)
            return new_full_dir_path
        except FileExistsError:
            pass
        count += 1

    raise FileExistsError(
        f"Unable to create a new directory with full path {full_dir_path} even"
        f" after {try_limit} changes to name"
    )
