import utilities.utils_tomo as utils 

def get_volume(projections, angles, centre=None, pad=0, algorithm='GRIDREC', iterations=1):
    """
    Reconstructions function. For algorithms can use GRIDREC, SIRT, or FBP. It returns a 3D stack. 
    """
    projections = utils.pad(projections, (pad,0))
    if centre ==None:
        centre_of_rotation=utils.findCentre(projections)
        print("centre_of_rotation",centre_of_rotation)
    else:
        centre_of_rotation = centre + pad
    volume = utils.tomo_recon(projections, angles, centre=centre_of_rotation, algorithm=algorithm, iterations=iterations)
    utils.volume_zero_edges(volume, pad)
    return volume

def get_reprojections(projections, angles, centre, pad, algorithm='GRIDREC', iterations=1):
    """
    This works by first reconstructing a 3D volume, then reprojecting it with ASTRA into projections.
    """
    volume = get_volume(projections, angles, centre, pad, algorithm=algorithm, iterations=iterations)
    if pad > 0:
        reprojections = utils.get_reprojections(volume, angles)[:,:,pad:-pad]
    else:
        reprojections = utils.get_reprojections(volume, angles)[:,:,:]
    return reprojections