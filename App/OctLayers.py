import numpy as np
import pandas as pd
import logging
import os
import fnmatch
import re
import xml.etree.ElementTree

import readIowaData
import readBioptigenOct
import readCirrusOct

logger = logging.getLogger(__name__)

class OctLayers(object):
    """Class to hold the surface information in xml files from the Iowa reference algorithms
    Usage : oct = OctLayers(filename = <path to surface file>,
                            [center_filename = <path to Grid_Center file>],
                            [laterality = 'OD' | 'OS'],
                            [raw_filename = <path to raw .img file>])
    If laterality is not provided we will try to autodetect it from the xml file or from the filename
    """
    #surface_labels = ['ILM','RNFL-GCL','GCL-IPL','IPL-INL','INL-OPL',
                      #'OPL-HFL','BMEIS','IS/OSJ','IB_OPR','IB_RPE','OB_RPE']
    #structure to hold surface label information
    
    
    def __init__(self, data=None, *args, **kargs):
        """Initialise the object"""
        # Define some class variables
        self.filename = None        # full path to the iowa surfaces xml file
        self.center_filename = None # full path to the iowa grid_centers xml file
        
        self.data = None            # numpy masked array holding the iowa surface coordinate information
                                    #  [nLayers x bscans x ascans] in pixels
        self.etdrs = None           # numpy array holding mask for edtrs regions
                                    #  [bscans x ascans]

        self.octdata = None         # OCT scan data
                                    #  [ascans x depth x bscans]

        self.center_x = None        # int, holding x coordinate of the scan center from grid_centers.xml
        self.center_y = None        # int, holding y coordinate of the scan center from grid_centers.xml
        self.laterality = None      # eye ('OD'|'OS)
        self.scan_size = None       # Dict {x,y,z} size of scan in pixels
        self.voxel_size = None      # Dict {x,y,z} size of voxels in microns

        if 'system' in kargs.keys():
            self.system = kargs['system'].lower()
        else:
            self.system = 'cirrus'
            
        #if not data:
            #if self.system == 'cirrus':
                #self.data = np.empty((11,128,512),dtype=np.float)
            #elif self.system == 'bioptigen':
                #self.data = np.empty((11,100,1000),dtype=np.float)
            #else:
                #raise ValueError('Invalid system')
        #else:
            #self.data = np.array(data)

    
        if data:
            self.data = np.array(data)
                   
        if 'filename' in kargs.keys():
            self.filename = kargs['filename']
            self.loadXml()
            
        if 'center_filename' in kargs.keys():
            self.center_filename = kargs['center_filename']
            self.loadCenter()
        else:
            self.center_x = None
            self.center_y = None

        if 'raw_filename' in kargs.keys():
            self.raw_filename = kargs['raw_filename']
            self.loadRawOct()

        self.etdrs = self.genEtdrsRings()
            
        # can override the autodetected laterality here
        if 'laterality' in kargs.keys():
            self.laterality = kargs['laterality']

    # delgation function
    def __getattr__(self,name):
        """Delegate to NumPy array"""
        try:
            return getattr(self.data, name)
        except:
            raise AttributeError(
                "'Array' object has no attribute {}".format(name))
    
    # delgation function
    def __getitem__(self,key):
        return self.data.__getitem__(key)
    
    # delgation function
    def __setitem__(self,key,value):
        self.data.__setitem__(key,value)    
            
    def loadXml(self):
        """Load the xml file defined in self.filename"""
        if not self.filename:
            logger.error('Source filename not set')
            raise IOError('Filename not set')
        
        logger.debug('Loading file:{}'.format(self.filename))
        results = readIowaData.readIowaSurfaces(self.filename)
        
        self.scan_size = results['scan_size']
        self.voxel_size = results['voxel_size']
        self.system = results['scan_system']
        self.laterality = results['eye']
        self.surface_labels = results['surface_names']
        self.data = results['surface_data']
                
    def loadCenter(self):
        """Load the xml file defined in self.center_filename"""
        if not self.center_filename:
            logger.error('Center filename not set')
            raise IOError('Filename not set')
        
        result = readIowaData.readIowaCenter(self.center_filename)
        self.center_x = result[0]
        self.center_y = result[1]
        
    def loadRawOct(self):
        """Load the xml file defined in self.raw_filename"""
        if self.raw_filename is None:
            logger.error('Raw filename not set')
            raise IOError('Filename not set')
        
        if self.system == 'cirrus' or self.raw_filename.lower().endswith('img'):
            octData = readCirrusOct.readCirrusOct(self.raw_filename)
        elif self.system == 'bioptigen' or self.raw_filename.lower().endswith('oct'):
            octData = readBioptigenOct.readBioptigenOct(self.raw_filename)
            
       
        self.octdata = octData['image_data']
        
    def centerData(self):
        """data matrix is shifted so that defined coordinates are in the center of the array
        Out of bounds areas are marked as np.NAN and are masked"""
        
        assert self.center_x and self.center_y, 'Center coords not set'
              
        # center point is defined as x,y coordinates in the original macular cube,
        # calculate how far to shift these coordinates to move them to the array center
        row_shift = self.center_y - int(self.data.shape[1] / 2)
        col_shift = self.center_x - int(self.data.shape[2] / 2)
        
        # perform the actual shift
        if row_shift <= 0:
            # shift data down
            t_idx = 0
            b_idx = self.data.shape[1]  - abs(row_shift)
            target_t_idx = abs(row_shift)
        else:
            # shift data up
            t_idx = row_shift
            b_idx = self.data.shape[1]
            target_t_idx = 0
            
        if col_shift <= 0:
            # shift data right
            l_idx = 0
            r_idx = self.data.shape[2] - abs(col_shift)
            target_l_idx = abs(col_shift)
        else:
            # shift data right
            l_idx = col_shift
            r_idx = self.data.shape[2]
            target_l_idx = 0
            
            
        # perform the centering
        if self.data is not None:
            # setup masked array data structures to hold the shifted data
            centered_data = np.ma.MaskedArray(np.empty(self.data.shape))
            centered_data[:] = np.NAN

            # copy valid datapoints into the new masked array
            # invalid datapoints are those moved from outside the original scanned area
            valid_data = self.data[:,t_idx:b_idx, l_idx:r_idx]        
            centered_data[:,
                          target_t_idx:(target_t_idx + valid_data.shape[1]),
                          target_l_idx:(target_l_idx + valid_data.shape[2])] = \
                valid_data
            # update the masked array to mark invalid datapoints
            centered_data.mask = (centered_data.mask + np.isnan(centered_data.data))
            # update the object with the new data
            self.data = centered_data

        if self.etdrs is not None:
            # setup array data structures to hold the shifted data
            centered_etdrs = np.empty(self.etdrs.shape)
            centered_etdrs[:] = np.NAN 
        
            # copy valid datapoints into the new masked array
            # invalid datapoints are those moved from outside the original scanned area        
            valid_etdrs = self.etdrs[t_idx:b_idx, l_idx:r_idx]
            centered_etdrs[target_t_idx:(target_t_idx + valid_data.shape[1]),
                           target_l_idx:(target_l_idx + valid_data.shape[2])] = \
                valid_etdrs
    
        if self.octdata is not None:
            #setup masked array data structures to hold the shifted data
            centered_octdata = np.ma.MaskedArray(np.empty(self.octdata.shape))
            centered_octdata[:] = np.NAN 
        
            # copy valid datapoints into the new masked array
            # invalid datapoints are those moved from outside the original scanned area        
            valid_octdata = self.octdata[t_idx:b_idx, :, l_idx:r_idx]
            centered_octdata[target_t_idx:(target_t_idx + valid_octdata.shape[0]),
                             :,
                             target_l_idx:(target_l_idx + valid_octdata.shape[2])] = \
                valid_octdata
            # update the masked array to mark invalid datapoints
            centered_octdata.mask = (centered_octdata.mask + np.isnan(centered_octdata.data))
            self.octdata = centered_octdata
    
    def setOrient(self,target='OD'):
        """Modify the autodetected orientation of the recording"""
        assert self.laterality, 'Original laterality not defined'
        if self.laterality != target:
            self.data = self.data[:,:,::-1]
            self.laterality = target
    
    def getThickness(self, surface1, surface2, mask = True):
        """Calculates the thickness between two surfaces
        return values are in microns"""
        thickness = np.ma.MaskedArray(np.abs(self.data.data[surface1,:,:] - self.data.data[surface2,:,:]))
        thickness = thickness * self.voxel_size['z']
        if surface1 == 0 and surface2 == 10:
            # getting total thickness, no need to mask center?
            mask = False
        
        if mask:
            thickness.mask = self.data.mask[0,:,:]
        else:
            thickness.mask = np.isnan(thickness.data)
            
        return thickness
    
    def genEtdrsRings(self):
        """generate a mask template for etdrs regions"""
        if self.voxel_size:
            #if we have loaded from xml file we can go this route
            pixSizeX = self.voxel_size['x'] / 1000
            pixSizeY = self.voxel_size['y'] / 1000
            
            scanSizeX = self.scan_size['x'] * pixSizeX
            scanSizeY = self.scan_size['y'] * pixSizeY
        else:
            # otherwise we have to guess a bit
            # TODO: Fix init so this can be gleaned from a source object if we are copying
            if self.system == 'cirrus':
                scanSizeX = float(6) # scan size in mm
                scanSizeY = float(6)
            elif self.system == 'bioptigen':
                #Bioptigen 12x12 scan
                scanSizeX = float(12) # 12 mm scan size
                scanSizeY = float(12)
            else:
                raise ValueError('Other OCT systems not yet implemented')
            
            pixSizeX = scanSizeX / self.data.shape[2]
            pixSizeY = scanSizeY / self.data.shape[1]            
            
        radius_mm = np.array([1,3,6],dtype=np.float) / 2
        radius_pix_x = radius_mm * pixSizeX
        radius_pix_y = radius_mm * pixSizeY
        
        y,x = np.ogrid[:self.data.shape[1],:self.data.shape[2]]
        x = x + 0.5 # move grid from pixel vertices to pixel centers
        y = y + 0.5
        x_mm = x * pixSizeX
        y_mm = y * pixSizeY
        
        cx_mm = scanSizeX / 2
        cy_mm = scanSizeY / 2
        
        # convert from cartesian to polar coordinates
        r2 = (x_mm-cx_mm)**2 + (y_mm-cy_mm)**2
        theta = np.arctan2(y_mm-cy_mm,x_mm-cx_mm)
        
        # define the quadrants
        qI=np.logical_and(theta>(np.pi/4), theta<(np.pi*(3.0/4)))
        qS=np.logical_and(theta>(-np.pi*(3.0/4)), theta<(-np.pi*(1.0/4)))
        qL=abs(theta)>np.pi*(3.0/4)
        qR=abs(theta)<np.pi*(1.0/4)

        # ring 1 mask
        r1Mask = np.zeros(r2.shape,dtype=np.bool)
        r1Mask[np.logical_and(r2 > radius_mm[0]**2, r2 < radius_mm[1]**2)] = 1

        # ring 2 mask
        r2Mask = np.zeros(r2.shape,dtype=np.bool)
        r2Mask[np.logical_and(r2 > radius_mm[1]**2, r2 < radius_mm[2]**2)] = 1
        
        etdrsMask = np.zeros(r2.shape,dtype=np.int)
        etdrsMask[r2 < radius_mm[0]**2] = 1 # fovea
        
        etdrsMask[np.logical_and(r1Mask,qI)] = 2 # Inferior inner
        etdrsMask[np.logical_and(r1Mask,qS)] = 3 # Superior inner
        etdrsMask[np.logical_and(r1Mask,qL)] = 4 # Left inner
        etdrsMask[np.logical_and(r1Mask,qR)] = 5 # Right inner
        
        etdrsMask[np.logical_and(r2Mask,qI)] = 6 # Inferior outer
        etdrsMask[np.logical_and(r2Mask,qS)] = 7 # Superior outer
        etdrsMask[np.logical_and(r2Mask,qL)] = 8 # Left outer
        etdrsMask[np.logical_and(r2Mask,qR)] = 9 # Right outer
              
        return etdrsMask
    
    def getEtdrsIntensity(self, surface1, surface2, applyMask):
        """Calculate layer intensity values for etdrs regions"""
        layer_mask = self.getOctLayerMask(surface1, surface2, applyMask)
        # make the 2D etdrs mask into a 3d cube same shape as layer_mask
        etdrs_mask = self.etdrs[:, np.newaxis, :]
        etdrs_mask = np.repeat(etdrs_mask, layer_mask.shape[1], 1)
        
        output = []
        
        for iRegion in np.arange(1, 10):
            lr_mask = np.logical_and(layer_mask,etdrs_mask==iRegion) #layer / region mask
            output.append(np.mean(self.octdata[lr_mask]))
            
        # get inner ring values
        ring_mask = np.logical_or.reduce((self.etdrs==2,
                                          self.etdrs==3,
                                          self.etdrs==4,
                                          self.etdrs==5))
        ring_mask = ring_mask[:, np.newaxis, :]
        ring_mask = np.repeat(ring_mask, layer_mask.shape[1], 1)
        lr_mask = np.logical_and(layer_mask, ring_mask)  #layer / region mask
        output.append(np.mean(self.octdata[lr_mask]))
        
        # get outer ring values
        ring_mask = np.logical_or.reduce((self.etdrs==6,
                                          self.etdrs==7,
                                          self.etdrs==8,
                                          self.etdrs==9))
        
        ring_mask = ring_mask[:, np.newaxis, :]
        ring_mask = np.repeat(ring_mask, layer_mask.shape[1], 1)
        lr_mask = np.logical_and(layer_mask, ring_mask)  #layer / region mask
        output.append(np.mean(self.octdata[lr_mask]))
        
        # get macular values
        mac_mask = self.etdrs > 0
        mac_mask = mac_mask[:, np.newaxis, :]
        mac_mask = np.repeat(mac_mask, layer_mask.shape[1], 1)
        lmac_mask = np.logical_and(layer_mask, mac_mask)
        output.append(np.mean(self.octdata[lmac_mask]))
        
        # Swap everything to RE orientation
        if self.laterality == 'OS':
            output[3],output[4] = output[4],output[3]
            output[7],output[8] = output[8],output[7]
            
        regions={'Fovea':output[0],
                 'InnerInferior':output[1],
                 'InnerSuperior':output[2],
                 'InnerNasal':output[3],
                 'InnerTemporal':output[4],
                 'OuterInferior':output[5],
                 'OuterSuperior':output[6],
                 'OuterNasal':output[7],
                 'OuterTemporal':output[8],
                 'Inner':output[9],
                 'Outer':output[10],
                 'Macular':output[11]}
    
        regions = pd.Series(regions)
        return regions        
        
    def getEtdrsThickness(self, surface1, surface2, applyMask):
        """Calculate layer thickness values for edtrs regions"""
        layer = self.getThickness(surface1, surface2, applyMask)
        output = []
        for iRegion in np.arange(1, 10):
            value = np.mean(layer[self.etdrs==iRegion])
            output.append(value)
            
        # get inner ring values
        value = np.mean(layer[np.logical_or.reduce((self.etdrs==2,
                                                    self.etdrs==3,
                                                    self.etdrs==4,
                                                    self.etdrs==5))])
        output.append(value)
        
        # get outer ring values
        value = np.mean(layer[np.logical_or.reduce((self.etdrs==6,
                                                    self.etdrs==7,
                                                    self.etdrs==8,
                                                    self.etdrs==9))])
        output.append(value)
        
        # get macular values
        value = np.mean(layer[self.etdrs>0])
        output.append(value)        
        
        if self.laterality == 'OS':
            output[3],output[4] = output[4],output[3]
            output[7],output[8] = output[8],output[7]
            
        regions={'Fovea':output[0],
                 'InnerInferior':output[1],
                 'InnerSuperior':output[2],
                 'InnerNasal':output[3],
                 'InnerTemporal':output[4],
                 'OuterInferior':output[5],
                 'OuterSuperior':output[6],
                 'OuterNasal':output[7],
                 'OuterTemporal':output[8],
                 'Inner':output[9],
                 'Outer':output[10],
                 'Macular':output[11]}
        
        regions = pd.Series(regions)
        return regions
    
    def overlayLayers(self):
        """overlay surfaces onto raw oct images
        returns an np.array[bscans,depth,ascans]"""
        if self.data is None:
            raise RuntimeError('Surface data not loaded')
        if self.octdata is None:
            raise RuntimeError('Raw oct images not loaded')
        
        oct = np.copy(self.octdata)
        for surface in range(self.data.shape[0]):
            for frame in range(self.data.shape[1]):
                for i,v in enumerate(self.data[surface,frame,:]):
                    if v is not np.ma.masked:
                        oct[frame,v-2:v+2,i]=255
        return oct
    def getOctLayerMask(self, surface1, surface2, mask = True):
        """Apply the layer data to the raw oct images
        returns a masked array of the same size as self.octdata
        with points outside the layer masked"""
        if self.data is None:
            raise RuntimeError('Surface data not loaded')
        if self.octdata is None:
            raise RuntimeError('Raw oct images not loaded')
        mask = np.zeros(self.octdata.shape,
                        dtype=np.bool)
        # ensure we have the top surface first
        surfaces = np.array([surface1,surface2])
        surfaces.sort()
        for frame in range(self.data.shape[1]):
            for i,v in enumerate(self.data[surfaces[0],frame,:]):
                v2 = self.data[surfaces[1],frame,i]
                if v is not np.ma.masked and v2 is not np.ma.masked:
                    mask[frame,v:v2,i] = True
        return mask
    def findFovea(self):
        """Attempt to find coordinates of the fovea
        Do this by finding the thinnest point between 
        layers 0 (ILM) and 3 (IPL-INL)"""
        layer = self.getThickness(0,3,False)
        indices = np.where(layer == layer.min())
        # in case there is more than one min value lets estimate the center
        # first exclude and outlier pixels
        #indices = zip(*indices) # convert from x,y pairs to [(x),(y)]
        minpoint = (np.percentile(indices[0],25),
                    np.percentile(indices[1],25))
        maxpoint = (np.percentile(indices[0],75),
                    np.percentile(indices[1],75))
        badpoints_x = np.logical_or(indices[0] < minpoint[0],
                                    indices[0] > maxpoint[0])
        badpoints_y = np.logical_or(indices[1] < minpoint[1],
                                    indices[1] > maxpoint[1])
        goodpoints = np.logical_not(np.logical_or(badpoints_x,badpoints_y))
        indices = (indices[0][goodpoints],indices[1][goodpoints])
        centroid = (np.mean(indices[0]),np.mean(indices[1]))
        self.center_y, self.center_x = [int(x) for x in centroid]
        
class OctCollection(object):
    
    def __init__(self,folder = None, laterality='OD',nmax=None):
        #nmax is a switch to only process a subset of files for testing
        assert laterality in ['OD','OS'], 'Laterality mus be OS or OD'
        self.data = []
        self.laterality = laterality
        self.srcFolder = folder
        if folder:
            self.loadFolder(nmax)
        
    def loadFolder(self,nmax=None):
        assert self.srcFolder, 'Source folder not set'
        surface_files = []
        center_files = []
        for dirname, _, filenames in os.walk(self.srcFolder):
            for name in fnmatch.filter(filenames,'*Surfaces_Iowa.xml'):
                surface_files.append((dirname,name))
            for name in fnmatch.filter(filenames,'*GridCenter_Iowa.xml'):
                center_files.append((dirname,name))                
                
        if nmax is not None:
            # only process a subset for testing
            surface_files = surface_files[0:nmax]
            
        p = re.compile('(.*)_Surfaces_Iowa.xml')
        for surface_file in surface_files:
            basename = re.match(p, surface_file[1]).group(1)
            try:
                i = [i for i,v in enumerate(center_files) if re.search(basename,v[1])][0]
            except IndexError:
                i = None
            
            if i is not None:
                data = OctLayers(filename = os.path.join(surface_file[0], surface_file[1]),
                                 center_filename = os.path.join(center_files[i][0],center_files[i][1]))
                logger.debug('Centering data')
                data.centerData()
            else:
                logger.warning('GridCenter file not found for surface file:{}'.format(surface_file[1]))
                data = OctLayers(filename = os.path.join(surface_file[0], surface_file[1]))
                
            data.setOrient(self.laterality)
            self.data.append(data)
            
    def getLayerValues(self, surface1, surface2, value_type='mean'):
        assert self.data, 'No data loaded'
        assert value_type in ['mean','stdev'], 'Invalid value_type requested'
        layers = np.ma.MaskedArray(np.empty((len(self.data),
                                             self.data[0].shape[1],
                                             self.data[1].shape[2])))
        for idx in range(len(self.data)):
            layer = self.data[idx].getThickness(surface1, surface2)
            layers[idx,:,:] = layer
        
        if value_type == 'mean':
            return layers.mean(axis=0)
        
        if value_type == 'stdev':
            return layers.std(axis=0)
        
    def setOrient(self, laterality='OD'):
        for recording in self.data:
            recording.setOrient(laterality)
            
    def getEtdrs(self, surface1, surface2, applyMask=True):
        output = pd.DataFrame()
        for recording in self.data:
            data = recording.getEtdrsThickness(surface1, surface2, applyMask)
            output = output.append(data, ignore_index=True)
        return output