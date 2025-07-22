import numpy as np
from skimage.registration import phase_cross_correlation as register_translation

def pad(projections_in, pad):
    npad = ((0, 0), (pad[1], pad[1]), (pad[0], pad[0]))
    projections_out = np.pad(projections_in, npad, mode='edge')
    return projections_out

def findCentre(projections):
    firstImage = projections[0,:,:]
    lastImage = projections[-1,:,:]
    firstImageFlipped = np.fliplr(firstImage)
    b,c=np.shape(firstImage)
    shift, error, diffphase = register_translation(firstImageFlipped,lastImage, upsample_factor=100)
    mycent=c/2-shift[1]/2
    return mycent

def volume_zero_edges(volume, width):
    ap = np.abs(genCircularApertureMask(np.shape(volume)[-2:], np.shape(volume)[-1]/2-width))
    ap = ap[np.newaxis, ...]
    np.multiply(volume, ap, out=volume)

# From ptyrex.core.toolbox
def genCircularApertureMask(shape, r=0, cent=None):
    """### Generate a circular aperture mask ###
    out:
        Array indicating which values fall inside of the aperture.
    in:
        shape - Shape of the source array, must be 2-dimensional.
        r     - Radius of the masking disc. Defaults to 0.
        cent  - None or the index on which to centre the disk.
                If None, defaults to the centre of the soure array.
    """

    if len(shape) != 2:
        raise ValueError('shape must be 2-dimensional')

    if cent is not None and len(cent) != len(shape):
        raise ValueError('shape and cannot have different dimensionality')

    # By default, mask everything, but bail if the radius is negative
    if r == 0:
        return np.zeros(shape, dtype=complex)
    elif r < 0:
        raise ValueError("cannot create a mask with negative radius")

    # If the aperture's center is undefined, default to the middle of the area
    if cent is None:
        cent = np.divide(np.array(shape) - 1, 2)

    y = np.arange(shape[0]) - cent[0]
    x = np.arange(shape[1]) - cent[1]

    # Cast boolean array of everything inside the circle to complex
    # so masking can happen by multiplication
    xs, ys = np.meshgrid(x, y)
    return np.array((xs ** 2 + ys ** 2 <= r ** 2), dtype=complex)