#cython: language_level=3, cdivision=True
from PIL import Image
import os

import numpy as np
cimport numpy as np
ctypedef np.uint8_t DTYPE_UINT8

from libc.stdlib cimport malloc, free


cdef class label_handler:

    cdef int *city_train_label
    cdef int *mapillary_label

    cdef void setupCityScapes(self, int igr):
        self.city_train_label = <int *> malloc ( 256 * sizeof(int))
        cdef int *label_arr = self.city_train_label

        label_arr[0] = igr
        label_arr[1] = igr
        label_arr[2] = igr
        label_arr[3] = igr
        label_arr[4] = igr
        label_arr[5] = igr
        label_arr[6] = igr
        label_arr[7] = 0
        label_arr[8] = 1
        label_arr[9] = igr
        label_arr[10] = igr
        label_arr[11] = 2
        label_arr[12] = 3
        label_arr[13] = 4
        label_arr[14] = igr
        label_arr[15] = igr
        label_arr[16] = igr
        label_arr[17] = 5
        label_arr[18] = igr
        label_arr[19] = 6
        label_arr[20] = 7
        label_arr[21] = 8
        label_arr[22] = 9
        label_arr[23] = 10
        label_arr[24] = 11
        label_arr[25] = 12
        label_arr[26] = 13
        label_arr[27] = 14
        label_arr[28] = 15
        label_arr[29] = igr
        label_arr[30] = igr
        label_arr[31] = 16
        label_arr[32] = 17
        label_arr[33] = 18
        label_arr[250] = igr

    #if class does not exist in cityscapes, it is not trained on
    cdef void setupMapillary(self, int igr):

        self.mapillary_label = <int *> malloc ( 256 * sizeof(int))
        cdef int *label_arr = self.mapillary_label

        label_arr[0] = igr      #bird
        label_arr[1] = igr      #animal
        label_arr[2] = igr      #curb
        label_arr[3] = 4    #fence
        label_arr[4] = igr      #guardrail
        label_arr[5] = igr      #other barriers
        label_arr[6] = 3    #wall
        label_arr[7] = 0    #bike lane (road)
        label_arr[8] = 0    #crosswalk (road)
        label_arr[9] = 0    #curb cut (road)
        label_arr[10] = 0   #parking (road)
        label_arr[11] = 1   #pedestrain area
        label_arr[12] = igr     #rail track
        label_arr[13] = 0   #road
        label_arr[14] = 0   #service lane (road)
        label_arr[15] = 1   #sidewalk
        label_arr[16] = igr     #bridge
        label_arr[17] = 2   #building
        label_arr[18] = igr     #tunnel
        label_arr[19] = 11  #person
        label_arr[20] = 12  #cyclist (rider)
        label_arr[21] = 12  #motorcyclist (rider)
        label_arr[22] = 12  #other rider
        label_arr[23] = igr     #lane marking -crosswalk
        label_arr[24] = igr     #lane marking - general
        label_arr[25] = 9   #mountain (terrain)
        label_arr[26] = 9   #sand (terrain)
        label_arr[27] = 10  #sky
        label_arr[28] = 9   #snow (terrain)
        label_arr[29] = 9   #terrain
        label_arr[30] = 8   #vegetation
        label_arr[31] = 9   #water (terrain)
        label_arr[32] = igr     #banner
        label_arr[33] = igr     #bench
        label_arr[34] = igr     #bike rack
        label_arr[35] = igr     #billboard
        label_arr[36] = igr     #catch-basin
        label_arr[37] = igr     #cctv camera
        label_arr[38] = igr     #fire hydrant
        label_arr[39] = igr     #junction box
        label_arr[40] = igr     #mailbox
        label_arr[41] = igr     #manhole
        label_arr[42] = igr     #phone booth
        label_arr[43] = igr     #pothole
        label_arr[44] = 5   #street light (pole)
        label_arr[45] = 5   #pole
        label_arr[46] = 7   #traffic sign frame (traffic sign)
        label_arr[47] = 5   #utility pole (pole)
        label_arr[48] = 6   #traffic light
        label_arr[49] = 7   #traffic sign back
        label_arr[50] = 7   #traffic sign front
        label_arr[51] = igr     #trash can
        label_arr[52] = 18  #bicycle
        label_arr[53] = igr     #boat
        label_arr[54] = 15  #bus
        label_arr[55] = 13  #car
        label_arr[56] = igr     #caravan
        label_arr[57] = 17  #motorcycle
        label_arr[58] = igr     #on rails
        label_arr[59] = igr     #other vehicle
        label_arr[60] = igr     #trailer
        label_arr[61] = 14  #truck
        label_arr[62] = igr     #wheeled slow
        label_arr[63] = igr     #car mount
        label_arr[64] = igr     #ego vehicle
        label_arr[65] = igr #unlabelled
        label_arr[250] = igr

    def __cinit__(self, int ignore_label):
        
        self.setupCityScapes(ignore_label)
        self.setupMapillary(ignore_label)
    
    def __dealloc__(self):
        """ make memory allocated available again """
        
        free(self.city_train_label)
        free(self.mapillary_label)
    
    cpdef np.ndarray[DTYPE_UINT8, ndim=2] label_cityscapes(self, np.ndarray[DTYPE_UINT8, ndim=2] mask):
        
        cdef int h, w, height, width

        height, width = np.shape(mask)

        for h in range(height):
            for w in range(width):
                
                mask[h, w] = self.city_train_label[ mask[h, w] ]
        
        return(mask)
    
    cpdef np.ndarray[DTYPE_UINT8, ndim=2] label_mapillary(self, np.ndarray[DTYPE_UINT8, ndim=2] mask):
        
        cdef int h, w, height, width

        height, width = np.shape(mask)

        for h in range(height):
            for w in range(width):
                
                mask[h, w] = self.mapillary_label[ mask[h, w] ]
        
        return(mask)