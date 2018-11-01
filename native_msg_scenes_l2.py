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


import os, sys
import logging
from datetime import datetime,timedelta
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
        print ('self1',filename_info)
        self.platform_name=filename_info['satellite']
        self.subsat=filename_info['subsat']
        print (filename_info['start_time'])
        self.rc_start=filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(self._get_memmap(), chunks=(CHUNK_SIZE,))
        
    @property
    def start_time(self):
        start_time=self.rc_start
        return start_time
        
    @property
    def end_time(self):
        end_time=self.rc_start+timedelta(minutes=15)
        return end_time
        
    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        lrec = [
                ('SceneType', np.uint8),
                ('QualityFlag', np.uint8)
                ]

        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', lrec, (self.mda['number_of_columns'],))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        return dt
    
    
    def _get_memmap(self):
        """Get the memory map for the SEVIRI data"""

        with open(self.filename) as fp:

            data_dtype = self._get_data_dtype()
            
            return np.memmap(fp, dtype=data_dtype,
                             shape=(self.mda['number_of_lines'],),
                             offset=self.hdr_size, mode="r")

    def _read_header(self):
        """Read the header info"""
        header_record = [
                     ('mpef_product_header', MpefHeader.mpef_product_header),
                     ('image_structure', ImageNavigation.image_structure),
                     ('image_navigation_noproj',
                      ImageNavigation.image_navigation_noproj)
                     ]

        self.header = np.fromfile(self.filename,
                           dtype=header_record, count=1)
        
        self.hdr_size= np.dtype(header_record).itemsize 
        
        
        #self.platform_id = int(self.header['mpef_product_header']['SpacecraftName'])
        #self.platform_name = "Meteosat-" + str(self.platform_id)
        
        print (self.platform_name)
        print (self.subsat)
        
        #~ if self.header['mpef_product_header']['Mission'].astype('|U3') == 'E0000':
            #~ ssp_lon=0.0
        #~ elif self.header['mpef_product_header']['Mission'].astype('|U3') == 'E0415':
            #~ ssp_lon=41.5
        #~ else:
            #~ ssp_lon=9.5
            
        if self.subsat == 'E0000':
            ssp_lon=0.0
        elif self.subsat == 'E0415':
            ssp_lon=41.5
        elif self.subsat == 'E0095':
            ssp_lon=9.5
        else:
             ssp_lon=0.0                

        self.mda['projection_parameters'] = {'a': '6378169.0',
                                             'b': '6356583.8',
                                             'h': 35785831.00,
                                             'ssp_longitude': ssp_lon}


        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] =  self.header['image_structure']['NoColumns'][0]
        
        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']

    
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

        center_point = self.mda['number_of_lines']/2

        north = self.mda['number_of_lines']
        east = self.mda['number_of_columns']
        west = 0
        south = 0

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
        raw = self.dask_array['DataRecord']['LineRecord']['SceneType']
        data=raw[:,0,:].flatten()
        data = da.flipud(da.fliplr((data.reshape(shape))))
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
	    

