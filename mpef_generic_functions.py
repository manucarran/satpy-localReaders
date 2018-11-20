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

# This file contains generic functions to be used by MPEF MSG Binary products
from pyresample import geometry
import numpy as np

def get_area_extent(dsid):
        
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
        
class mpefGenericFuncs(object):

    def get_area_def(dsid,nlines,ncols,lon_0):

        a = 6378169.0
        b = 6356583.8
        h = 35785831.00
        
        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            get_area_extent(dsid))

        return area

    def get_memmap(filename,data_dtype,noLines,hdrSize):
        """Get the memory map for the SEVIRI data"""
        print ("!!!!!!!!!!!!!!!!!!!!!IN GENERIC MMAP!!!!!!!!!!!!!!!!!!!!!!")
        with open(filename) as fp:

            #data_dtype = self._get_data_dtype()
            
            return np.memmap(fp, data_dtype,
                             shape=(noLines,),
                             offset=hdrSize, mode="r")
       
    def read_header(hdr_rec,filename):
        """Read the header info"""
        header = np.fromfile(filename,
                                  dtype=hdr_rec, count=1)

        hdr_size = np.dtype(hdr_rec).itemsize

 
        header = header.newbyteorder('>')
        nlines = header['image_structure']['NoLines'][0]
        ncols = header['image_structure']['NoColumns'][0]
        
        return nlines,ncols
        
