from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.morphology import ball, erosion


def _save_tiff(data: np.ndarray, output_file: Path) -> None:
    """Save image data to a TIFF file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_file), data)


def save_segmentation(
    bw: np.ndarray,
    contour_flag: bool,
    output_path: Path,
    fn: str,
    suffix: str = "_struct_segmentation",
):
    """Save the segmentation into a tiff file

    Parameters
    ----------
    bw: np.ndarray
        the segmentation to save
    contour_flag: book
        whether to also save segmentation contour
    output_path: Path
        the path to save
    fn: str
        the core file name to use, for example, "img_102", then
        after a suffix (say "_seg") is added, the file name of the output
        is "img_101_seg.tiff"
    suffix: str
        the suffix to add to the output filename
    """
    _save_tiff(bw, output_path / (fn + suffix + ".tiff"))

    if contour_flag:
        bd = generate_segmentation_contour(bw)

        _save_tiff(bd, output_path / (fn + suffix + "_contour.tiff"))


def generate_segmentation_contour(im):
    """Generate the contour of the segmentation"""
    bd = np.logical_xor(erosion(im > 0, footprint=ball(1)), im > 0)

    bd = bd.astype(np.uint8)
    bd[bd > 0] = 255

    return bd


def output_hook(im, names, out_flag, output_path, fn):
    """General hook for cutomized output"""
    assert len(im) == len(names) and len(names) == len(out_flag)

    for i in range(len(out_flag)):
        if out_flag[i]:
            if names[i].startswith("bw_"):
                segmentation_type = names[i]
                bw = im[i].astype(np.uint8)
                bw[bw > 0] = 255
                _save_tiff(
                    bw, output_path / (fn + "_bw_" + segmentation_type[3:] + ".tiff")
                )
            else:
                _save_tiff(im[i], output_path / (fn + "_" + names[i] + ".tiff"))
