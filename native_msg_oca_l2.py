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
from datetime import datetime,timedelta
import numpy as np

import xarray as xr
import dask.array as da

from satpy import CHUNK_SIZE
from pyresample import geometry

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.eum_base import recarray2dict
from .mpef_definitions import GlobalTypes, ImageNavigation,  MpefHeader

from .mpef_generic_functions import  mpefGenericFuncs

logger = logging.getLogger('native_msg_ndvi')

class MSGOcaFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGOcaFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name=filename_info['satellite']
        self.subsat=filename_info['subsat']
        self.rc_start=filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))
        #self.dask_array = da.from_array(self._get_memmap(), chunks=(CHUNK_SIZE,))
        
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
        self.parameters = {
        'cot': 'ULTau',
        'cot2': 'LLTau',
        'cph': 'Phase',
        'cre': 'ULRe',
        'ctp': 'ULCtp',
        'ctp2': 'LLCtp',
        'ecot': 'ULSnTau',
        'ecot2': 'LLSnTau',
        'ectp': 'ULSnCtp',
        'ectp2': 'LLSnCtp',
        'ecre': 'ULSnRe',
        'jm': 'Jm'
        }
        
        drec = [
            ('Latitude', np.float32),
            ('Longitude', np.float32),
            ('QualityFlag', np.float32),
            ('Phase', np.float32),
            ('Jm', np.float32),
            ('ULTau', np.float32),
            ('ULCtp', np.float32),
            ('ULRe', np.float32),
            ('ULSnTau', np.float32),
            ('ULSnCtp', np.float32),
            ('ULSnRe', np.float32),
            ('LLTau', np.float32),
            ('LLCtp', np.float32),
            ('LLSnTau', np.float32),
            ('LLSnCtp', np.float32),
            ]
            
        
        dt = np.dtype([('DataRecord', drec, (3712,))])
        dt = dt.newbyteorder('>')
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
                     ]

        self.header = np.fromfile(self.filename,
                           dtype=header_record, count=1)
        
        self.hdr_size= np.dtype(header_record).itemsize 
            
        if self.subsat == 'E0000':
            self.ssp_lon=0.0
        elif self.subsat == 'E0415':
            self.ssp_lon=41.5
        elif self.subsat == 'E0095':
            self.ssp_lon=9.5
        else:
             self.ssp_lon=0.0                

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] =3712
        self.mda['number_of_columns']=3712
        
        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']
        
    def get_area_def(self, dsid):
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid,nlines,ncols,self.ssp_lon)
        
    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = self.parameters[dsid.name]
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['{}'.format(channel)]
        data=raw[:,:]
        
        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).astype(np.float32)
        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            if dsid.name=='cot' or dsid.name=='cot2':
                dataset=10**dataset
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'
        
        return dataset	        
