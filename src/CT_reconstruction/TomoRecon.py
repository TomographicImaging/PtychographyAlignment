import utilities.utils_tomo as utils
import utilities.utils_used as utils_used

def get_volume(projections, angles, centre=None, pad=0, algorithm='GRIDREC', iterations=1):
    """
    Reconstructions function. For algorithms can use GRIDREC, SIRT, or FBP. It returns a 3D stack. 
    """
    projections = utils_used.pad(projections, (pad,0))
    if centre ==None:
        centre_of_rotation=utils_used.findCentre(projections)
        print("centre_of_rotation",centre_of_rotation)
    else:
        centre_of_rotation = centre + pad
    volume = tomo_recon(projections, angles, centre=centre_of_rotation, algorithm=algorithm, iterations=iterations)
    utils_used.volume_zero_edges(volume, pad)
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

def tomo_recon(projections, angles, centre=None, algorithm='FBP', iterations=1):
    # Currently recast_tomo is not in the environment, but we are only using tomopy.
    if centre == None:
        centre = utils_used.findCentre(projections)
    if algorithm=='FBP':
        try:
            from recast_tomo import FBP2D_ASTRA
        except ImportError as e:
            print(e)
        volume=FBP2D_ASTRA(projections, angles, centre)
    elif algorithm=='SIRT':
        try:
            from recast_tomo import SIRT2D_ASTRA
        except ImportError as e:
            print(e)
        volume=SIRT2D_ASTRA(projections, angles, centre, iterations)
    elif algorithm=='GRIDREC':
        try:
            import tomopy
        except ImportError as e:
            print(e)
        volume = tomopy.recon(projections, angles, algorithm='gridrec', center=centre)
    return volume