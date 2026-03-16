# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:32:37 2025

@author: vdz11526
"""
import sys
import os
# sys.path.append(os.path.abspath(".."))
import h5py
import numpy as np
import matplotlib.pyplot as plt
from archive import tomoconsistency_tools_oriol as tc
from archive import tomoconsistency_tools_hannah as tch
# from VerticalAlignmentSwiss import VerticalAlignmentSwiss as va
# from utilities import utils_tomo
from scipy.signal import windows 
from scipy.ndimage import convolve
from scipy.ndimage import center_of_mass
import time
#%%
# file = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.nxs'
file = '/dls/i13-1/data/2025/cm40629-1/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.mat'
data_key = '/stack_object'

with h5py.File(file, 'r') as f:
    img_orig = np.angle(f[data_key][:,:,:])

theta = np.linspace(0,np.pi,img_orig.shape[-1])
theta

vert_crop = 75
horiz_crop = 75

[Ny, Nx, Nangles] = img_orig.shape
object_ROI = [np.ceil(np.arange(1+vert_crop, Ny-vert_crop)), 
              np.ceil(np.arange(1+horiz_crop, Nx-horiz_crop))]

# Make data easily splitable for ASTRA, preferable size of blocks should be dividable by 32
width_sinogram = np.ceil(len(object_ROI[1])/32)*32
Nlayers = np.floor(len(object_ROI[0])/32)*32
Nroi = [len(object_ROI[0]),len(object_ROI[1])]

object_ROI = [object_ROI[0][int(np.ceil(Nroi[0]/2))]+np.arange(-Nlayers/2,Nlayers/2),
              object_ROI[1][int(np.ceil(Nroi[1]/2))]+np.arange(-width_sinogram/2,width_sinogram/2)]

Npix = np.ceil(1.0*width_sinogram/32)*32  # for pillar it can be the same as width_sinogram
vert_range = np.arange(32,Nlayers-33) # selected vertical layers for alignment 

#%%

img_orig_grad = tc.get_phase_gradient_1D(img_orig,ax=1)[int(object_ROI[0][0]):int(object_ROI[0][-1]+1),int(object_ROI[1][0]):int(object_ROI[1][-1]+1),:]

width_sinogram = img_orig_grad.shape[1]
high_pass_filter = 0.01
unwrap_data_method = 'fft_1d'

# include the effect of high pass filter into the weights 
size = np.maximum(3, int(np.ceil(high_pass_filter * width_sinogram)))
gauss_window = windows.gaussian(size, std = size/6)
hanning_window = windows.hann(3)
ker = gauss_window.reshape(-1,1) * hanning_window

# relevance weights -> remove effect of potential residues / phase jumps 
ker2 = ker[np.newaxis,:,:]
convolution_result = convolve((np.abs(img_orig_grad) > 2).astype(np.float32), ker2.astype(np.float32), mode = 'constant', cval = 0.0)
weights_find_shift = np.maximum(0,1-convolution_result)
# weights = windows.tukey(sinogram.shape[1], alpha= 0.2)

shift_total = np.zeros((img_orig_grad.shape[-1],2))
#%% crosscorrelation and vertical alignment results from MATLAB code
shifttt = np.zeros((181,2))
horiz_shifts = "34	53	71	53	46	60	49	59	94	53	49	42	16	10	2	-1	3	7	-7	-10	-12	16	-10	-10	15	23	12	16	-1	7	1	20	5	-15	-13	-20	-43	-32	-31	-22	-28	-41	-30	-17	-10	-14	5	16	1	8	-3	11	9	2	-10	-14	-4	1	-3	-12	0	2	4	23	-4	17	17	15	-3	16	13	-8	1	20	24	12	-4	12	21	26	26	17	39	37	22	10	33	38	37	25	28	25	34	37	36	25	20	19	14	21	6	29	11	5	12	18	23	22	4	14	5	2	-7	-17	4	10	5	-23	-7	-16	-5	-14	-15	-12	-30	-2	8	37	9	-13	-25	-1	-20	-38	-12	-42	-38	-30	-14	-50	-29	-18	-39	-16	-18	-24	-30	-17	-9	-5	-18	-8	-18	-21	-27	-14	-34	-17	-8	-1	11	15	33	50	45	32	55	43	52	71	65	55	78	82	86	81	88	87	102	99"
verti_shifts = "15.413752	18.869938	17.952105	15.048748	15.317489	16.457272	12.412716	12.346809	10.830973	11.337815	10.017136	5.8283844	10.083050	5.8169851	6.0416551	4.2934666	5.6583614	5.9371600	4.0354943	5.7538233	3.3005543	3.7650852	5.2703390	1.8591213	5.4564781	2.8896022	2.1997428	-0.17467356	-1.9699535	0.87871838	-3.2357435	0.76823330	-2.5602198	-2.1836128	-1.9600611	-4.5140209	-0.36251450	-3.0791235	-4.6122513	-9.9342089	-7.4991922	-5.3061371	-6.2287035	-5.0426693	-5.3855629	-7.9935055	-7.9635601	-7.0324230	-7.1765413	-6.2677298	-8.1166430	-9.2110443	-12.414388	-11.159865	-10.753753	-8.2610178	-13.170260	-11.250447	-10.745543	-14.356794	-11.571807	-11.877501	-12.783373	-13.843271	-14.373517	-10.941634	-14.464982	-12.502694	-13.538865	-12.422956	-11.257554	-13.799178	-10.734569	-10.189597	-12.750818	-9.7716684	-11.705909	-9.7415876	-11.212492	-10.909909	-12.516102	-14.289781	-8.9605494	-10.871581	-12.549095	-11.684525	-11.340709	-8.3016386	-9.5872774	-9.6175232	-4.7576141	-8.8119392	-7.6817207	-5.5180302	-4.1780596	-6.6357822	-6.3888206	-9.7493811	-7.3984365	-8.9965677	-4.9184933	-6.0479937	-6.1219254	-8.5863094	-4.4894323	-6.1594181	-5.1547985	-1.6491451	-2.7983265	-4.8097448	-5.3291540	0.41888046	-2.4759121	-4.6141396	-0.88960218	2.6166501	-2.0870996	-0.31212139	-2.6427526	-2.8247013	-1.2003055	-1.7152777	0.36425114	2.0175438	4.4538631	5.2944336	5.1687856	6.4964113	7.2222686	6.8367939	8.5167551	4.5047450	6.3769894	6.7455845	6.9169655	10.386265	8.1206379	8.1449308	10.452344	11.728489	9.8030329	10.273876	13.346823	12.488131	9.9812059	10.827835	15.084057	11.828949	19.741692	16.388168	16.768448	17.337341	15.531812	16.133751	15.302582	18.366550	17.546852	22.554668	17.627645	20.540474	22.700542	20.932266	21.294281	23.177410	25.027546	26.459126	23.972595	27.682163	26.186676	22.967850	26.703648	26.649876	28.135220	29.545984	28.637554	31.443890	31.050766	25.562595	29.057617	29.617386"
shifttt[1:,0] = np.fromstring(horiz_shifts, sep=' ')
shifttt[1:,1] = np.fromstring(verti_shifts, sep=' ')

tomoconsistency = "-37.7277259826660	-37.8945159912109	-38.7943801879883	-39.0557479858398	-39.7978324890137	-39.8872184753418	-40.7743034362793	-40.7953453063965	-42.1077919006348	-41.9481773376465	-42.1066856384277	-42.7708663940430	-44.3620834350586	-45.6523590087891	-44.9542846679688	-45.2062721252441	-45.5518569946289	-45.8271255493164	-46.8677330017090	-47.1818923950195	-47.5107917785645	-48.5594253540039	-47.7099533081055	-48.4170303344727	-48.8548355102539	-49.9091873168945	-50.3903808593750	-50.9284324645996	-51.7970046997070	-52.5053443908691	-52.5534820556641	-53.4966125488281	-53.2671737670898	-54.0382118225098	-54.1395759582520	-54.5419998168945	-54.3408317565918	-54.4794540405273	-55.5260200500488	-54.6670417785645	-55.6048431396484	-55.8875350952148	-55.4641571044922	-55.7226486206055	-55.7416267395020	-56.1917457580566	-55.1783790588379	-55.2039489746094	-55.0976676940918	-54.8440971374512	-54.4020614624023	-54.7839317321777	-54.0909004211426	-53.5370063781738	-54.2194290161133	-53.3242874145508	-53.2442703247070	-53.3538093566895	-52.6577339172363	-51.2443275451660	-49.9987869262695	-50.0194625854492	-48.3622894287109	-47.9628791809082	-47.6327400207520	-46.4836006164551	-46.1039123535156	-46.6966094970703	-45.7606697082520	-45.8875236511231	-46.3629684448242	-46.0228385925293	-46.3167037963867	-45.4254417419434	-46.4284820556641	-46.1835403442383	-46.7315635681152	-46.8060798645020	-47.4454460144043	-47.5009613037109	-48.0318489074707	-47.0685653686523	-47.3415336608887	-46.8325080871582	-47.7262535095215	-48.0869483947754	-47.8868522644043	-48.0267677307129	-48.0072326660156	-48.3421363830566	-48.9948005676270	-49.2590293884277	-49.7466316223145	-49.9278678894043	-51.0731277465820	-50.9506835937500	-51.6758270263672	-51.9698905944824	-53.0187263488770	-53.0940208435059	-54.6542625427246	-54.6646080017090	-54.3517189025879	-56.8218116760254	-55.9670257568359	-58.2666511535645	-59.2793769836426	-59.4508743286133	-59.5721549987793	-59.9975204467773	-61.4341125488281	-60.9468612670898	-61.7026062011719	-62.2899093627930	-63.8165855407715	-63.7266387939453	-64.4806442260742	-65.1445922851563	-66.2178497314453	-66.8173294067383	-66.7024002075195	-66.8408966064453	-66.9965515136719	-67.7221221923828	-68.3193206787109	-69.3304748535156	-69.7073516845703	-70.2348022460938	-70.6217651367188	-71.0488586425781	-70.6780624389648	-71.8713912963867	-72.5244369506836	-73.0625915527344	-72.5677413940430	-73.9616317749023	-74.6633148193359	-74.2915115356445	-74.4043731689453	-73.9481811523438	-75.5868988037109	-74.8687667846680	-75.6461410522461	-75.6809539794922	-76.0517120361328	-76.6483535766602	-77.4743957519531	-77.0904769897461	-77.5171585083008	-77.5788726806641	-78.2168807983398	-77.3568954467773	-78.0747070312500	-77.3106765747070	-76.3547363281250	-76.3382110595703	-77.0891799926758	-78.4100494384766	-76.7751998901367	-77.3059082031250	-77.8300857543945	-77.7954254150391	-79.1620712280273	-78.7266540527344	-79.3798065185547	-79.1569747924805	-80.0777130126953	-80.5688858032227	-80.7656707763672	-80.7777862548828	-80.3507843017578	-81.3205032348633	-80.6903991699219	-80.0404434204102	-79.5144500732422	-79.5489730834961	-78.7515411376953	-78.0769958496094	-77.3538055419922	-78.2353439331055"
tomoconsistency_shifts = np.fromstring(tomoconsistency, sep=' ')

for m in range(shifttt.shape[0]):
    img_orig_grad[:,:,m] = np.roll(img_orig_grad[:,:,m],(int(shifttt[m,0]), int(shifttt[m,1])),axis=(1,0)) 
#%%
import tomoconsistency_tools_oriol as tc

sinogram = img_orig_grad[int(vert_range[0])-1:int(vert_range[-1])+1,:,:].copy()
weights_find_shift = np.ones_like(sinogram)
high_pass_filter = 0.001
unwrap_data_method = 'fft_1d'
shift_method = 'geometry' # physical or geometry

# binning
binning = [8, 4, 2, 1] 
max_iteration = 100
dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
weights = np.full(len(theta), dtheta, dtype=np.float32)
plot_figures = True
min_step_size = 0.05 #max_update * par.binning < par.min_step_size

Npix = []

optimal_shift = np.zeros((sinogram.shape[-1],2))

for b in range(len(binning)):
    
    optimal_shift, shift_history = tc.align_tomo_consistency_linear(sinogram, weights_find_shift, weights, theta, Npix, optimal_shift, binning[b], 
                                                                    high_pass_filter = high_pass_filter, unwrap_data_method = 'fft_1d')
    
    shift_history = np.array(shift_history)
    plt.figure(figsize=(10,5))
    
    # for i in range(ii):
    #     plt.plot(theta, shift_history[i, :, 0], color='blue', alpha=0.3, label='x')
    #     # plt.plot(theta, shift_history[i, :, 1], color='red', alpha=0.3, label='y')
    #     plt.xlabel("Angle")
    #     plt.ylabel("Shift value")
    # plt.plot()

    plt.plot(theta, optimal_shift[:,0])
    plt.ylabel('Horizontal shift value')
    plt.xlabel('Angle (deg)')
    plt.legend(['Calculated shifts'])
    plt.grid()




#%% limiting the vertical range

# binning
binning = 8
sinogram = tc.imshift_generic(img_orig_grad[int(vert_range[0])-1:int(vert_range[-1])+1,:,:], shift_total, Npix = None, affine_matrix = None, smooth = 0, 
                                      ROI = None, downsample = binning, interp_method = 'linear', interp_sign = 0)

weights_find_shift = tc.imshift_generic(weights_find_shift[int(vert_range[0])-1:int(vert_range[-1])+1,:,:], shift_total, Npix = None, affine_matrix = None, smooth = 0, 
                                      ROI = None, downsample = binning, interp_method = 'linear', interp_sign = 0)

sinogram = tc.unwrap_data(sinogram, 'fft_1d', boundary=None)
Nlayers = sinogram.shape[0]

# ASTRA needs the reconstruction to be dividable by 32 othewise there
# will be artefacts in left corner 
Npix = np.ceil(Npix/binning);
if np.isscalar(Npix):
    Npix = [Npix, Npix, Nlayers];
if len(Npix) == 2:
    Npix = [Npix, Nlayers];


#%%
iteration_no = 5

Nx = sinogram.shape[1]
Ny = sinogram.shape[0]
Nangles = sinogram.shape[2]

vol_geom, proj_geom = tch.init_astra(Nx, Ny, theta)

dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
weights = np.full(len(theta), dtheta, dtype=np.float32)

#%%   
#### tomoconsistency
center_reconstruction = False
plot_figures = False

for ii in range(iteration_no):
    t0 = time.time()
    # shift with imdeform_affine_fft
    sinogram_shifted = tch.imshift_fft(sinogram, shift_total)

    if plot_figures:
        plt.figure()
        plt.subplot(121),plt.imshow(sinogram[:,:,0]),plt.colorbar()
        plt.subplot(122),plt.imshow(sinogram_shifted[:,:,0]),plt.colorbar()

    # fbp (ASTRA needs shape Ny * Nangle * Nx)
    sinogram_shifted = sinogram_shifted.transpose((0, 2, 1)) # for astra
    
    cor = tc.find_cor(sinogram_shifted, first=5)
    sinogram_shifted = np.roll(sinogram_shifted,int(cor),axis=2)
    
    rec = tch.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights)
    
    rec_mask = tch.apply_circular_mask(rec, 0.9)
    rec_mask = np.maximum(0,rec_mask)
    
    if plot_figures: 
        plt.figure()
        plt.subplot(131),plt.imshow(rec_mask[:,:,rec.shape[2]//2]),plt.colorbar()
        plt.subplot(132),plt.imshow(rec_mask[:,rec.shape[1]//2,:]),plt.colorbar()
        plt.subplot(133),plt.imshow(rec_mask[rec.shape[0]//2,:,:]),plt.colorbar()
    
    if center_reconstruction:
        # centering 
        rec_center = tc.centering_reconstruction(rec_mask)
        
        if ii == 0:
            if center_reconstruction:
                rec_center_0 = [0,0]
            else:
                rec_center_0 = rec_center
            
        shift_rec = -0.5*(rec_center - rec_center_0)
        
        rec_mask = tc.imshift_fft(rec_mask,shift_rec[0],shift_rec[1])
    
    # get reprojection
    sinogram_model = tch.get_projections(rec_mask, vol_geom, proj_geom)
    
    sinogram_shifted = sinogram_shifted.transpose((0, 2, 1))
    sinogram_model = sinogram_model.transpose((0, 2, 1))
    
    if plot_figures:
        min_c = np.min(sinogram_shifted[:,:,0])
        max_c = np.max(sinogram_shifted[:,:,0])
        plt.figure()
        plt.subplot(121),plt.imshow(sinogram_shifted[:,:,0],vmin=min_c,vmax=max_c),plt.colorbar()
        plt.subplot(122),plt.imshow(sinogram_model[:,:,0],vmin=min_c,vmax=max_c),plt.colorbar()
  
    MASS = np.median(sinogram_shifted * np.mean(abs(sinogram_shifted), axis=(0,1)))
    
    # sinogram_model is reprojected sinogram
    # sinogram is the original sino (also called "sinogram_shifted" in the MATLAB code)
    shift_upd, err = tc.find_optimal_shift(sinogram_model, sinogram_shifted, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
    
    step_relaxation = 0.01
    shift_upd = np.minimum(0.5, abs(shift_upd))*np.sign(shift_upd)*step_relaxation
    
    shift_total = shift_total + shift_upd
    plt.figure()
    plt.plot(shift_total[:,0], 'r', label='Total x shift')
    plt.plot(shift_total[:,1], 'b', label='Total y shift')
    plt.plot(shift_upd[:,0], '--r', label='Latest x shift')
    plt.plot(shift_upd[:,1], '--b', label='Latest y shift')
    plt.ylim([-0.02, 0.02])
    plt.legend()

    print(f'Iteration {str(ii)} time {time.time()-t0}')

if plot_figures is False: 
    plt.figure()
    plt.subplot(131),plt.imshow(rec_mask[:,:,rec_mask.shape[2]//2])
    plt.subplot(132),plt.imshow(rec_mask[:,rec_mask.shape[2]//2,:])
    plt.subplot(133),plt.imshow(rec_mask[rec_mask.shape[2]//2,:,:])
    plt.tight_layout()
    

