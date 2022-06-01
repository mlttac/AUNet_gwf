import ctypes as ct
import numpy as np
import random as rand
import os
from multiprocessing import Pool

def rand2d(n, max_lvl, cellsize, theta, xcells, ycells, seed, mean, sd, lognormal, aniso=1):

    DLL = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2D.dll'))
    squashstretch = DLL.__rand2dmod_MOD_squashstretch
    canda = DLL.__rand2dmod_MOD_canda
    rand2dDLL = DLL.__rand2dmod_MOD_rand2d

    max_lvl_ptr = ct.pointer(ct.c_int(max_lvl))
    level = max_lvl
    level_ptr = ct.pointer(ct.c_int(level))

    squash = 1
    squash_ptr = ct.pointer(ct.c_int(squash))
    #squash_ptr = np.ctypeslib.as_ctypes(squash)

    stretch = 1
    stretch_ptr = ct.pointer(ct.c_int(stretch))

    cellsize_ptr = ct.pointer(ct.c_double(cellsize))

    theta_ptr = ct.pointer(ct.c_double(theta))

    xcells_ptr = ct.pointer(ct.c_int(xcells))

    ycells_ptr = ct.pointer(ct.c_int(ycells))

    seed_ptr = ct.pointer(ct.c_int(seed)) #ct.c_int(-rand.randint(1, abs(seed))))

    meantop = mean
    meantop_ptr = ct.pointer(ct.c_double(meantop))

    mean_ptr = ct.pointer(ct.c_double(mean))
    sd_ptr = ct.pointer(ct.c_double(sd))

    aniso_ptr = ct.pointer(ct.c_int(aniso))

    lognormal_ptr = ct.pointer(ct.c_bool(lognormal))

    field = np.zeros([ycells, xcells])
    field_ptr = np.ctypeslib.as_ctypes(field)

    a_9c = np.zeros([3, 9, max_lvl])
    a_9c_ptr = np.ctypeslib.as_ctypes(a_9c)

    c_9c = np.zeros([6, max_lvl])
    c_9c_ptr = np.ctypeslib.as_ctypes(c_9c)

    squashstretch(aniso_ptr, max_lvl_ptr, xcells_ptr, ycells_ptr, level_ptr, squash_ptr, stretch_ptr)
    canda(a_9c_ptr, c_9c_ptr, level_ptr, cellsize_ptr, theta_ptr)

    while True:
        fields = []
        for realisation in range(n):
            #seed = seed - 100
            print(seed_ptr.contents)
            rand2dDLL(xcells_ptr, ycells_ptr, field_ptr, a_9c_ptr, c_9c_ptr, level_ptr, cellsize_ptr, seed_ptr, theta_ptr,
                      meantop_ptr, mean_ptr, sd_ptr, lognormal_ptr, squash_ptr, stretch_ptr)
            fields.append(np.array(np.transpose(field)))
        yield fields

def mainRFCall(args):
    max_lvl, xcells, ycells, cellsize, theta, field, seed, mean, sd, lognormal, aniso = args
    DLL = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RF2D.dll'))
    squashstretch = DLL.__rand2dmod_MOD_squashstretch
    canda = DLL.__rand2dmod_MOD_canda
    rand2dDLL = DLL.__rand2dmod_MOD_rand2d
    
    max_lvl_ptr = ct.pointer(ct.c_int(max_lvl))

    cellsize_ptr = ct.pointer(ct.c_double(cellsize))

    xcells_ptr = ct.pointer(ct.c_int(xcells))

    ycells_ptr = ct.pointer(ct.c_int(ycells))

    seed_ptr = ct.pointer(ct.c_int(-rand.randint(1, abs(seed))))

    meantop = mean
    meantop_ptr = ct.pointer(ct.c_double(meantop))

    mean_ptr = ct.pointer(ct.c_double(mean))
    sd_ptr = ct.pointer(ct.c_double(sd))

    lognormal_ptr = ct.pointer(ct.c_bool(lognormal))

    field_ptr = np.ctypeslib.as_ctypes(field)
    
    rand_aniso = rand.randint(1, aniso)
    aniso_ptr = ct.pointer(ct.c_int(rand_aniso))
    
    squash = 1
    squash_ptr = ct.pointer(ct.c_int(squash))
    
    stretch = 1
    stretch_ptr = ct.pointer(ct.c_int(stretch))
    
    level = max_lvl
    level_ptr = ct.pointer(ct.c_int(level))
            
    squashstretch(aniso_ptr, max_lvl_ptr, xcells_ptr, ycells_ptr, level_ptr, squash_ptr, stretch_ptr)
    rand_theta = rand.random() * theta
    theta_in = squash_ptr.contents.value * rand_theta

    theta_ptr = ct.pointer(ct.c_double(theta_in))
    
    #print ("Adjust: ", theta, theta_in, rand_aniso)
    
    a_9c = np.zeros([3, 9, max_lvl])
    a_9c_ptr = np.ctypeslib.as_ctypes(a_9c)

    c_9c = np.zeros([6, max_lvl])
    c_9c_ptr = np.ctypeslib.as_ctypes(c_9c)
    
    canda(a_9c_ptr, c_9c_ptr, level_ptr, cellsize_ptr, theta_ptr)
    
    rand2dDLL(xcells_ptr, ycells_ptr, field_ptr, a_9c_ptr, c_9c_ptr, level_ptr, cellsize_ptr,
              seed_ptr, theta_ptr, meantop_ptr, mean_ptr, sd_ptr, lognormal_ptr, squash_ptr, stretch_ptr)
              
    return np.array(np.transpose(field)), rand_theta, rand_aniso

def rand2d_theta(n, max_lvl, cellsize, theta, xcells, ycells, seed, mean, sd, lognormal, aniso=1):
    field = np.zeros([ycells, xcells])
    while True:
        #for realisation in range(n):
        #    fields.append(mainRFCall(max_lvl, max_lvl_ptr, xcells_ptr, ycells_ptr, cellsize_ptr, theta, field, field_ptr, seed_ptr, meantop_ptr, mean_ptr, sd_ptr, lognormal_ptr, squashstretch, canda, rand2dDLL))
        #yield fields
        p = Pool(8)
        args = [max_lvl, xcells, ycells, cellsize, theta, field, seed, mean, sd, lognormal, aniso]
        args_in = [args for _ in range(n)]
        fields, theta_out, aniso_out = zip(*(p.map(mainRFCall, args_in)))
        yield fields, theta_out, aniso_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 8           # number of realisations in one set
    max_lvl = 10     # number of levels of subdivision (2**max_lvl) is size.
    cellsize = 0.5  #
    theta = 10.0
    xcells = 100
    ycells = 75
    seed = -26021981
    mean = 10.0
    sd = 0.5
    lognormal = True
    aniso = 60.0
        
    rf2D = rand2d_theta(n, max_lvl, cellsize, theta, xcells, ycells, seed, mean, sd, lognormal, aniso);
    # everytime you call this you need a new seed number.
    fields, theta_out, aniso_out = next(rf2D)
    fields, theta_out, aniso_out = next(rf2D)
    print(fields[0].shape, theta_out, aniso_out)

    plt.subplot(1, 3, 1)
    plt.imshow(fields[0].transpose())
    plt.title('Field 1')
    plt.subplot(1, 3, 2)
    plt.imshow(fields[1].transpose())
    plt.title('Field 2')
    plt.subplot(1, 3, 3)
    plt.imshow(fields[2].transpose())
    plt.title('Field 3')
    plt.show()

    # maxK = 1
    # minK = 0.1
    # n_intervals = 5
    # RF = rand2d(1, 7, 64 / (64 - 1), 32, 64,64, seed, 0.0, 1.0, False, 1)
    # k = next(RF)
    # k = np.array(k)
    # k = np.interp(k, (k.min(), k.max()), (minK, maxK))
    # intervals = np.linspace(start=0.1, stop=1, num=n_intervals+1)
    # k = np.where(k >= intervals[-2], maxK, k)
    # k = np.where(k <= intervals[1], minK, k)
    # for index, interval in enumerate(intervals[1:-1]):
    #     k = np.where((k >= intervals[index]) & (k <= intervals[index+1]), intervals[index], k)
    # k = np.around(k,1)
    # print( np.unique(k))
    # im = plt.imshow(k[0])
    # plt.colorbar(im)
