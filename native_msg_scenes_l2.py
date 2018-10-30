#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Colin Duff <colin.duff@external.eumetsat.int>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import logging
from datetime import datetime
import numpy as np

import xarray as xr
import dask.array as da

from satpy import CHUNK_SIZE
from pyresample import geometry

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.eum_base import recarray2dict
from .mpef_definitions import GlobalTypes, ImageNavigation,  MpefHeader


logger = logging.getLogger('native_msg_scenes')

class MSGScenesFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGScenesFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(self._get_memmap(), chunks=(CHUNK_SIZE,))
    
    @property
    def start_time(self):
        start_time="20181010000000Z"
        return start_time
        #return self.header['15_DATA_HEADER']['ImageAcquisition'][
        #    'PlannedAcquisitionTime']['TrueRepeatCycleStart']

    @property
    def end_time(self):
        end_time="20181010000000Z"
        return end_time
        #return self.header['15_DATA_HEADER']['ImageAcquisition'][
        #    'PlannedAcquisitionTime']['PlannedRepeatCycleEnd']
	    
    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        lrec = [
                ('SceneType', np.uint8),
                ('QualityFlag', np.uint8)
                ]

        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', lrec, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        return dt
    
    
    def _get_memmap(self):
        """Get the memory map for the SEVIRI data"""

        with open(self.filename) as fp:

            data_dtype = self._get_data_dtype()
            hdr_size = 196
            print ('header',hdr_size)
            print (self.mda['number_of_lines'])
            print (data_dtype.itemsize)
            return np.memmap(fp, dtype=data_dtype,
                             shape=(self.mda['number_of_lines'],),
                             offset=hdr_size, mode="r")

    def _read_header(self):
        """Read the header info"""
        header_record = [
                     ('mpef_product_header', MpefHeader.mpef_product_header),
                     ('image_structure', ImageNavigation.image_structure),
                     ('image_navigation_noproj',
                      ImageNavigation.image_navigation_noproj)
                     ]

        data = np.fromfile(self.filename,
                           dtype=header_record, count=1)
        
        print (data)
        #self.header.update(recarray2dict(data)
        self.header=data 
        
        #data15hd = self.header['15_DATA_HEADER']
        #sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']

        # Set the list of available channels:
        
        self.platform_id = str(self.header['mpef_product_header']['SpacecraftName'].astype('|S2'))
        print (self.platform_id)
        self.platform_name = "Meteosat-" + self.platform_id
        print ('name',self.platform_name)
        
        print (self.header['mpef_product_header']['Mission'].astype('|U3'))
        if self.header['mpef_product_header']['Mission'].astype('|U3') == 'FES':
            ssp_lon=0.0
        
        #Projection: {'a': '6378169.0', 'b': '6356583.8', 'h': '35785831.0', 'lon_0': '0.0', 'proj': 'geos', 'units': 'm'}

        self.mda['projection_parameters'] = {'a': '6378169.0',
                                             'b': '6356583.8',
                                             'h': 35785831.00,
                                             'ssp_longitude': ssp_lon}

        west = 3712
        east = 3712
        # We suspect the UMARF will pad out any ROI colums that
        # arent divisible by 4 so here we work out how many pixels have
        # been added to the column.
        x = ((west - east + 1) * (10.0 / 8) % 1)
        y = int((1 - x) * 4)

        if y < 4:
            # column has been padded with y pixels
            cols_visir = int((west - east + 1 + y) * 1.25)
        else:
            # no padding has occurred
            cols_visir = int((west - east + 1) * 1.25)


        self.mda['number_of_lines'] = 3712
        self.mda['number_of_columns'] = 3712

        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']
        #ncols_hdr = int(sec15hd['NumberLinesVISIR']['Value'])

        #if ncols != ncols_hdr:
        #    logger.warning(
        #        "Number of VISIR columns from header and derived from data are not consistent!")
        #    logger.warning("Number of columns read from header = %d", ncols_hdr)
        #    logger.warning("Number of columns calculated from data = %d", ncols)

    
    def get_area_def(self, dsid):

        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['ssp_longitude']

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            self.get_area_extent(dsid))

        return area

    def get_area_extent(self, dsid):

        
        if dsid.name != 'HRV':

            # following calculations assume grid origin is south-east corner
            # section 7.2.4 of MSG Level 1.5 Image Data Format Description
            # origins = {0: 'NW', 1: 'SW', 2: 'SE', 3: 'NE'}
            # grid_origin = data15hd['ImageDescription'][
            #     "ReferenceGridVIS_IR"]["GridOrigin"]
            # if grid_origin != 2:
            #     raise NotImplementedError(
            #         'Grid origin not supported number: {}, {} corner'
            #         .format(grid_origin, origins[grid_origin])
            #     )

            center_point = 3712/2

            north = 3712
            east = 3712
            west = 0
            south =0

            # column_step = data15hd['ImageDescription'][
            #     "ReferenceGridVIS_IR"]["ColumnDirGridStep"] * 1000.0
            # line_step = data15hd['ImageDescription'][
            #     "ReferenceGridVIS_IR"]["LineDirGridStep"] * 1000.0
            # # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
            # earth_model = data15hd['GeometricProcessing']['EarthModel'][
            #     'TypeOfEarthModel']
            # if earth_model == 2:
            #     ns_offset = 0  # north +ve
            #     we_offset = 0  # west +ve
            # elif earth_model == 1:
            #     ns_offset = -0.5  # north +ve
            #     we_offset = 0.5  # west +ve
            # else:
            #     raise NotImplementedError(
            #         'unrecognised earth model: {}'.format(earth_model)
            #     )

            # section 3.1.5 of MSG Level 1.5 Image Data Format Description
            # ll_c = (center_point - west - 0.5 + we_offset) * column_step
            # ll_l = (south - center_point - 0.5 + ns_offset) * line_step
            # ur_c = (center_point - east + 0.5 + we_offset) * column_step
            # ur_l = (north - center_point + 0.5 + ns_offset) * line_step
            area_extent =  (-5570248.477339745, -5567248.074173927, 5567248.074173927, 5570248.477339745)
            #area_extent = (ll_c, ll_l, ur_c, ur_l)

        return area_extent

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])

        print ('data record',self.dask_array['DataRecord'])
        raw = self.dask_array['DataRecord']['LineRecord']['SceneType']
        data=raw[:,0,:]
        data = da.flipud(da.fliplr((data.reshape(shape))))
        print (data)
        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)

        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset	        
	    

