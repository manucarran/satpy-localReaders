

# Copyright (c) 2017 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Modified by John Jackson <john.jackson@external.eumetsat.int>

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

"""Nowcasting SAF MSG NetCDF4 format reader
"""

import logging
from datetime import datetime
from datetime import timedelta

import h5netcdf
import numpy as np
import xarray as xr

from pyresample.utils import get_area_def
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import proj_units_to_meters
from satpy import CHUNK_SIZE
logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'MET-08': 'Meteosat-8',
                  'MET-09': 'Meteosat-9',
                  'MET-10': 'Meteosat-10',
                  'MET-11': 'Meteosat-11', }


class NcOCAFileHandler(BaseFileHandler):

    """MSG OCA NetCDF reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(NcOCAFileHandler, self).__init__(filename, filename_info,
                                          filetype_info)
        
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'number_tie_points_act': CHUNK_SIZE,
                                          'number_tie_points_alt': CHUNK_SIZE})

#        import pdb; pdb.set_trace()        
        self.nc.attrs['x']= int(self.nc.dims['number_tie_points_act'])
        self.nc.attrs['y']= int(self.nc.dims['number_tie_points_alt'])
	
        self.nc = self.nc.rename({'number_tie_points_act': 'x', 'number_tie_points_alt': 'y'}) 

        # the processing_time is taken from the filename, however could be taken from remote_sensing_start_time if/when attribute is set properly.
        self.processing_time = filename_info['year']+'-'+filename_info['month']+'-'+filename_info['day']+'T'+filename_info['hour']+':'+filename_info['min']+':00Z'
#        print(self.nc)

        self.sensor = self.nc.attrs['instrument']
        sat_id = str(self.nc.attrs['spacecraft'])

        self.mda={}
        self.mda['projection_parameters'] = {'a': '6378169.0',
                                             'b': '6356583.8',
                                             'h': 35785831.00,
                                             'ssp_longitude': '0.0'}
        self.mda['number_of_lines'] = self.nc.attrs['y']
        self.mda['number_of_columns'] =self.nc.attrs['x']
        
        try:
            self.platform_name = PLATFORM_NAMES[sat_id]
        except KeyError:
            self.platform_name = PLATFORM_NAMES[sat_id.astype(str)]

    def get_dataset(self, dataset_id, dataset_info):
        
        dataset = self.nc[dataset_info['nc_key']]
        dataset.attrs.update(dataset_info)	
	
        # Correct for the scan line order (N->S) and (W->E)
        dataset = dataset.sel(y=slice(None, None, -1))
        dataset = dataset.sel(x=slice(None, None, -1))	
        dataset.attrs['_FillValue'] = 0
        return dataset


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

#        area_extent = (float(self.nc.attrs['gdal_xgeo_up_left']),
#                       float(self.nc.attrs['gdal_ygeo_low_right']),
#                       float(self.nc.attrs['gdal_xgeo_low_right']),
#                       float(self.nc.attrs['gdal_ygeo_up_left']))         

        area_extent =  (-5570248.477339745, -5567248.074173927, 5567248.074173927, 5570248.477339745)

        area = get_area_def('some_area_name',
                            "On-the-fly area",
                            'geosmsg',
                            proj_dict,
                            ncols,
                            nlines,
                            area_extent)

        return area

    @property
    def start_time(self):
        try:	
	   
             return datetime.strptime(self.processing_time,
                                      '%Y-%m-%dT%H:%M:%SZ')	     

        except TypeError:
	        
	        return datetime.strptime(
                self.processing_time.astype(str),
                                     '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        try:

            minutes = timedelta(minutes=15)
            return datetime.strptime(self.processing_time,
                                     '%Y-%m-%dT%H:%M:%SZ')+minutes
        except TypeError:
            
            return datetime.strptime(
                self.processing_time.astype(str),
                                     '%Y-%m-%dT%H:%M:%SZ')+minutes


