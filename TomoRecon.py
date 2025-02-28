import utils_tomo as utils 
# Reconstructions function. For algorithms can use GRIDREC, SIRT, or FBP. It returns a 3D stack. 
def get_volume(projections, angles, centre=None, pad=0, algorithm='GRIDREC', iterations=1):
    projections = utils.pad(projections, (pad,0))
    if centre ==None:
        centre_of_rotation=utils.findCentre(projections)
    else:
        centre_of_rotation = centre + pad
    volume = utils.tomo_recon(projections, angles, centre=centre_of_rotation, algorithm=algorithm, iterations=iterations)
    utils.volume_zero_edges(volume, pad)
    return volume