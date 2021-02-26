"""General utility functions """
import pathlib
import datetime
import os.path as osp
from typing import Union, List, NamedTuple


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


def save_results(all_results: List[NamedTuple],
                 result_dir: str,
                 filename: str,
                 include_timestamp: bool = True):
    """Save all results to file """
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        full_filename = osp.join(result_dir, f"{filename}_{timestamp}.csv")
    else:
        full_filename = osp.join(result_dir, f"{filename}_{timestamp}.csv")

    print(f"\nSaving results to: {full_filename}")

    with open(full_filename, "w") as fout:
        headers = all_results[0]._fields
        fout.write("\t".join(headers) + "\n")
        for result in all_results:
            row = [str(v) for v in result._asdict().values()]
            fout.write("\t".join(row) + "\n")


def append_result_to_file(results: Union[List[NamedTuple], NamedTuple],
                          filepath: str,
                          add_header: bool = True):
    """Append result to file

    Will add header if add_header is True and the file at filepath doesn't
    exist
    """
    if not isinstance(results, list):
        results = [results]

    if not osp.isfile(filepath) and add_header:
        with open(filepath, "w") as fout:
            headers = results[0]._fields
            fout.write("\t".join(headers) + "\n")

    with open(filepath, "a") as fout:
        for result in results:
            row = []
            for v in result._asdict().values():
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            fout.write("\t".join(row) + "\n")
