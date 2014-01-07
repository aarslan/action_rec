import sad
import numpy as np
import pylab as plt
import ipdb



##parameters
iscale   = 2
ndir     = 4
speed    = [1,2]
npatches = 1000
patchSizes = [8,16]
thre_for_sample_c1 = 0.3

#######################
## S1
# Template matching (spatio-temporal filters)
s1siz = [5,7,9,11,13,15,17,19,21,23,25,27]
#s1scales{1}    = 1:4
#s1scales{2}    = 5:8
#s1scales{3}    = 9:12

nf1const = 0.1

#s1Par = struct('Ssiz',s1siz(s1scales{iscale}),'ndir',ndir,'speed',speed,'symmetry',[0],'Tsiz',nt*ones(4,1),'nf1const',0.1,'filter_type','gabor','thres',0.2)

s1_templates = sad.gabor(shape=(10, 50, 50), scale=4, orientation=(0, np.pi/4, np.pi/2, (5*np.pi)/4), velocity = speed)

#######################
## C1
# Local max over each S1 map (tolerance to position)
#c1siz = [6,8,10,12,14,16]
#nt = [9];
#c1step = [3,4,5,6,7,8]
#overlap = nt-1
#c1scales{1} = 1:2
#c1scales{2} = 3:4
#c1scales{3} = 5:6
    
#c1stage_num = 1+4+4+4+1
#c1Par = struct('PoolSiz',c1siz(c1scales{iscale}),'PoolStepSiz',c1step(c1scales{iscale}),'PoolScaleSiz',2,'PoolScaleStepSiz',2)


with sad.ImageStream('c1') as stream:
    for frame in sad.window(sad.Video('Megamind.avi'),10):
        frame = np.mean(frame, axis=1)
        s1 = sad.match(frame, s1_templates)
        ipdb.set_trace()
        c1 = sad.pool(s1, (3,3), (6,6))
        ipdb.set_trace()
        stream.write(c1)

########################
### S2
## Template matching with d1 stored prototypes -> d1 S2 maps
#s2Par = struct('exponent',1,'thres',0.001)
#
#
########################
### C2
## Global max over each S2 map (invariance to position)
#c2stage_num = 1+4+4+4+2+2+1
#c2Par = struct('PoolSiz',200,'PoolStepSiz',1,'PoolScaleSiz',2,'PoolScaleStepSiz',1)


#######################
## S3
# Template matching to d2 stored prototypes -> d2 S3 maps



#######################
##C3
# Global max over each S3 map (invariance to shift in time)
