"""
This script provides the training samples for the neural network model.
It allows the user to define the boundaru head values and two hydraulic 
conductivities. It create the inputs for MODFLOW (via flowpy), runs the model 
and places the input and output files in compressed folders.

It is inspired by the example: 
https://github.com/modflowpy/flopy/blob/develop/examples/Notebooks/flopy3_mf6_A_simple-model.ipynb 

This script requires that `flopy` be installed within the Python
environment you are running this script in.
"""

import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import math
import pyRan2DMod as RF

# np.random.seed(260282)
# np.random.seed(260286)

def split_intervals(k, minK, maxK,n_intervals):
    k = np.interp(k, (k.min(), k.max()), (minK, maxK))
    intervals = np.linspace(start=minK, stop=maxK, num=n_intervals+1)
    k = np.where(k >= intervals[-2], maxK, k)
    k = np.where(k <= intervals[1], minK, k)
    for index, interval in enumerate(intervals[1:-1]):
        k = np.where((k >= intervals[index]) & (k <= intervals[index+1]), intervals[index], k)
    k = np.around(k,1)
    return k


class modflow6:

    def __init__(self, name, h_boundary, N, L, minK=None, maxK=None, n_intervals=2 , k=None, h_wells=None, wells_pos=None, n_wells=4):

        # print(sys.version)
        # print('numpy version: {}'.format(np.__version__))
        # print('matplotlib version: {}'.format(mpl.__version__))
        # print('flopy version: {}'.format(flopy.__version__))

        # For this example, we will set up a model workspace.
        # Model input files and output files will reside here.
        self.workspace = os.path.join('data', name)
        self.name = name

        self.h_boundary = h_boundary

        self.setUpWells = self.setupRandWells(h_wells, wells_pos, n_wells)

        self.N = N
        self.L = L

        self.minK = minK
        self.maxK = maxK
        self.n_intervals = n_intervals
        self.RF = None

        if k is None:
            self.randK = True
            self.setup_RF(32.0, -26021981)
        else:
            self.randK = False
            self.k = k

        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

    def setupRandWells(self, h_wells, wells_pos, n_wells):
        all_coords = [[x, y] for x in range(1, self.N-1) for y in range(1, self.N-1)]
        if h_wells is None and wells_pos is None:
            randWells = True
        else:
            randWells = False

        while True:
            if randWells:
                noWells = np.random.randint(1, n_wells)
                h_wells = []
                for i in range(noWells):
                    h_wells.append(np.random.uniform(0.5, 0.9) * self.h_boundary)
                wells_pos_ind = np.random.choice(len(all_coords), noWells, replace=False)
                wells_pos = [all_coords[i] for i in wells_pos_ind]
            else:
                h_wells = h_wells
                wells_pos = wells_pos

            yield h_wells, wells_pos

    def setup_RF(self, theta, seed):
        max_lvl = int(math.log(self.N)/math.log(2))+1
        #print("Max Level: {}".format(max_lvl))
        self.RF = RF.rand2d(1, max_lvl, self.L / (self.N - 1), theta, self.N, self.N, seed, 0.0, 1.0, False, 1)




    def run(self):
        """ Runs a simple 2D domain with wells and boundaries in Modflow 6 producing output files"""

        while True:

            if self.randK:
                input_arr = np.zeros((self.N, self.N, 3), dtype=float)
            else:
                input_arr = np.zeros((self.N, self.N, 2), dtype=float)

            output_arr = np.zeros((self.N, self.N, 1), dtype=float)

            if self.randK:
                self.k = next(self.RF)
                self.k = np.array(self.k)
                if self.n_intervals==2:
                    self.k = np.where(self.k >= 0, self.maxK, self.minK)
                else:
                    self.k = split_intervals(self.k, self.minK, self.maxK, self.n_intervals)
                    
                
                input_arr[:, :, 2] = self.k[0]


            h_wells, wells_pos = next(self.setUpWells)

            # Create the Flopy simulation object
            sim = flopy.mf6.MFSimulation(sim_name=self.name, exe_name='./mf6.2.1/mf6.2.1/bin/mf6.exe',
                                         version='mf6', sim_ws=self.workspace,
                                         verbosity_level=0)

            # Create the Flopy temporal discretization object
            tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, pname='tdis', time_units='DAYS', nper=1,
                                                        perioddata=[(1.0, 1, 1.0)])

            # Create the Flopy groundwater flow (gwf) model object
            model_nam_file = '{}.nam'.format(self.name)
            gwf = flopy.mf6.ModflowGwf(sim, modelname=self.name, 
                                       model_nam_file=model_nam_file, save_flows=True)

            # Create the Flopy iterative model solver (ims) Package object
            ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname='ims', complexity='SIMPLE')

            # Create the discretization package
            #delrow = delcol = self.L / (self.N - 1)
            delrow = delcol = 1 # width of every row and column
            dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(gwf, pname='dis', nrow=self.N, ncol=self.N, delr=delrow, delc=delcol)

            # Create the initial conditions package
            start = h1 * np.ones((1, self.N, self.N))  # Head constant over domain at 1
            ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname='ic', strt=start)

            # Create the node property flow package
            npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf, pname='npf', icelltype=1, k=self.k,
                                                           save_flows=True)

            # Create the constant head package.
            # List information is created a bit differently for
            # MODFLOW 6 than for other MODFLOW versions.  The
            # cellid (layer, row, column, for a regular grid)
            # must be entered as a tuple as the first entry.
            # Remember that these must be zero-based indices!

            # Well Head Position
            chd_rec = []
            for pos, well_head in zip(wells_pos, h_wells):
                chd_rec.append(((0, pos[0], pos[1]), well_head))

            # Boundary Head
            for row_col in range(0, N):
                chd_rec.append(((0, row_col, 0), self.h_boundary))
                chd_rec.append(((0, row_col, self.N - 1), self.h_boundary))
                if row_col != 0 and row_col != self.N - 1:
                    chd_rec.append(((0, 0, row_col), self.h_boundary))
                    chd_rec.append(((0, self.N - 1, row_col), self.h_boundary))

            chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf, pname='chd', maxbound=len(chd_rec),
                                                           stress_period_data=chd_rec, save_flows=True)

            # The chd package stored the constant heads in a structured
            # array, also called a recarray.  We can get a pointer to the
            # recarray for the first stress period (iper = 0) as follows.
            iper = 0
            ra = chd.stress_period_data.get_data(key=iper)

            # We can make a quick plot to show where our constant
            # heads are located by creating an integer array
            # that starts with ones everywhere, but is assigned
            # a -1 where chds are located
            ibd = np.zeros((1, self.N, self.N), dtype=int)
            ihd = h1 * np.ones((1, self.N, self.N))
            for [k, i, j], head in zip(ra['cellid'], ra['head']):
                ibd[k, i, j] = 1
                ihd[k, i, j] = head

            ilay = 0

            input_arr[:, :, 0] = ihd[ilay, :, :]
            input_arr[:, :, 1] = ibd[ilay, :, :]

            # Create the output control package
            headfile = '{}.hds'.format(self.name)
            head_filerecord = [headfile]
            budgetfile = '{}.cbc'.format(self.name)
            budget_filerecord = [budgetfile]
            saverecord = [('HEAD', 'ALL'),
                          ('BUDGET', 'ALL')]
            printrecord = [('HEAD', 'LAST')]
            oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf, pname='oc', saverecord=saverecord,
                                                        head_filerecord=head_filerecord,
                                                        budget_filerecord=budget_filerecord,
                                                        printrecord=printrecord)
            
            # Note that help can always be found for a package
            # using either forms of the following syntax
            #help(oc)
            #help(flopy.mf6.modflow.mfgwfoc.ModflowGwfoc)
            
            
            # Write the datasets
            sim.write_simulation()

            # Run the simulation
            success, buff = sim.run_simulation()

            if success:
                # Read the binary head file and plot the results
                # We can use the existing Flopy HeadFile class because
                # the format of the headfile for MODFLOW 6 is the same
                # as for previous MODFLOW verions
                fname = os.path.join(self.workspace, name + ".hds")
                hds = flopy.utils.binaryfile.HeadFile(fname)
                output_arr[:, :, 0] = hds.get_data(kstpkper=(0, 0))[0]
                
                
# #                 read the binary grid file
#                 fname = os.path.join(self.workspace, '{}.dis.grb'.format(name))
#                 bgf = flopy.mf6.utils.MfGrdFile(fname)
                
#                 # data read from the binary grid file is stored in a dictionary
#                 #bgf._datadict
                                
#                 # read the cell budget file
#                 fname = os.path.join(self.workspace, '{}.cbb'.format(name))
#                 cbb = flopy.utils.CellBudgetFile(fname, precision='double')
#                 #cbb.list_records()
                
#                 #Flow between two adjacent cells
#                 flowja = cbb.get_data(text='FLOW-JA-FACE')[0][0, 0, :] 
                
#                 #Flow between the groundwater system and a
#                 #constant-head boundary or a group of cells
#                 #with constant-head boundaries
#                 chdflow = cbb.get_data(text='CHD')[0]
#                 # By having the ia and ja arrays and the flow-ja-face we can look at
#                 # the flows for any cell and process them in the follow manner.
#                 k = 1; i = 1; j = 1
#                 celln = k * N * N + i * N + j
#                 celln = 60
#                 ia, ja = bgf.ia, bgf.ja
#                 print('Printing flows for cell {}'.format(celln))
#                 # TO DO! change: when the list decreases 
#                 for ipos in range(ia[celln] + 1, ia[celln + 1]):
#                     cellm = ja[ipos]
#                     print('Cell {} flow with cell {} is {}'.format(celln, cellm, flowja[ipos]))
#                     print('Ipos {}'.format(ipos))
#                
#                bname = os.path.join(self.workspace, name + ".cbb")
#                cbb = flopy.utils.binaryfile.CellBudgetFile(bname)
#                flowja = cbb.get_data(text="FLOW-JA-FACE", kstpkper=(0, 0))[0]
#                grb_file = "{}.dis.grb".format(name)
#                grb_file = os.path.join(self.workspace, name + ".dis.grb")
#                residual = flopy.mf6.utils.postprocessing.get_residuals(flowja, grb_file=grb_file)
#                #output_arr[:, :, 1] = cbb.get_data(kstpkper=(0, 0))[0]
#                
#                flopy.utils.postprocessing.get_extended_budget(cbb, kstpkper=(0, 0))[0]
#                flopy.utils.postprocessing.get_specific_discharge(cbb, gwf,hds)
#                flopy.utils.postprocessing.get_specific_discharge(gwf,cbb, hdsfile=hds, position='centers')
                yield input_arr, output_arr

            else:
                raise RuntimeError("Modflow failed to run")


def plot_IO(input, output, fixed):

    if not fixed:
        no = 5
    else:
        no = 3
    
    plt.figure(figsize=(15,15))
    
    count = 1
    ax1 = plt.subplot(1, no, count)
    im = ax1.imshow(input[:, :, 0])
    plt.colorbar(im,fraction=0.05, pad=0.03)
    count += 1
    ax2 = plt.subplot(1, no, count)
    im = ax2.imshow(input[:, :, 1])
    plt.colorbar(im,fraction=0.05, pad=0.03)
    count += 1
    if not fixed:
        ax3 = plt.subplot(1, no, count)
        im = ax3.imshow(input[:, :, 2])
        plt.colorbar(im,fraction=0.05, pad=0.03)
        count += 1

    ax4 = plt.subplot(1, no, count)
    im = ax4.imshow(output[:, :, 0])
    plt.colorbar(im,fraction=0.05, pad=0.03)
    plt.axis('scaled')
    plt.tight_layout()
    plt.show()
    plt.pause(3)


file_list = []
if __name__ == "__main__":
    for run in range (10): 
        name = 'mf6ai'
        outputName = "{}.npz".format(2+run)
        file_list.append(outputName)
        h1 = 1
        N = 64
        L = 64.0
        minK = 0.1
        maxK = 1
        n_intervals = 5 
        fixed = False
        visualPlot = False
        k = None
        if fixed:
            k = 1.0
    
        if True:
            print(k)
            mf6 = modflow6(name, h1, N, L, k=k, minK=minK, maxK= maxK, n_intervals= n_intervals).run() # minK=1, maxK=20
            realisations = [[10000, "Train"], [1000, "Test"]]
            x_train = []
            x_test = []
            y_train = []
            y_test = []
            folders = [[x_train, y_train], [x_test, y_test]]
    
            for realisation, folder in zip(realisations, folders):
    
                for iter in range(realisation[0]):
                    # print("\n\nRunning Iter: " + str(
                        # iter) + " " + realisation[1] + " Data\n=============================================================================")
                    input_arr, output_arr = next(mf6)
                    if visualPlot:
                        plot_IO(input_arr, output_arr, fixed)
                    folder[0].append(input_arr)
                    folder[1].append(output_arr)                   
    
            np.savez_compressed(outputName, x_train=np.array(x_train), y_train=np.array(y_train),
                                x_test=np.array(x_test), y_test=np.array(y_test))

#loaded = np.load(outputName)
#print(loaded['x_train'].shape)
#print(loaded['y_train'].shape)
#print(loaded['x_test'].shape)
#print(loaded['y_test'].shape)
#print(loaded['x_validation'].shape)
#print(loaded['y_validation'].shape)


# file_list = ['file_0.npz', 'file_1.npz', ...]
# file_list = [name, name, name]
data_all = [np.load(fname) for fname in file_list]
merged_data = {}
for data in data_all:
    [merged_data.update({k: v}) for k, v in data.items()]
np.savez('new_file.npz', **merged_data)


import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#count = 0
#while True:
#    plot_IO(loaded['x_train'][count], loaded['y_train'][count], fixed)
#    count += 1





