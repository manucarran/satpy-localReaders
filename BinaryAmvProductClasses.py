import os
import sys
import logging
from datetime import timedelta
import numpy as np
import xarray as xr
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.eum_base import recarray2dict
from satpy.readers.mpef_definitions import SegProdHeaders, SegProdDrecs

logger = logging.getLogger('BinaryAmvProductClasses')

sub_sat_dict = {"E0000": 0.0, "E0415": 41.5, "E0095": 9.5}

class MSGAmvIntFileHandler(BaseFileHandler):
    """Reader for MSG AMV Intermediate product in native format."""

    def __init__(self, filename, filename_info, filetype_info):
        super(MSGAmvIntFileHandler, self).__init__(filename, filename_info, filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat        = filename_info['subsat']
        self.rc_start      = filename_info['start_time']
        self.ssp_lon       = sub_sat_dict[filename_info['subsat']]

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start + timedelta(minutes=15)

    def _read_amv_header(self):
        """Read AMV product header."""
        hdr = np.fromfile(self.filename, SegProdHeaders.amvIntm_prd_hdr, 1)
        hdr = hdr.newbyteorder('>')
        return recarray2dict(hdr)

    def get_dataset(self, dataset_id, dataset_info):

        hdr_size = np.dtype(SegProdHeaders.amvIntm_prd_hdr).itemsize
        dtype    = np.dtype(SegProdDrecs.amvIntm)

        data = np.fromfile(self.filename, dtype=dtype, offset=hdr_size)
        data = data.newbyteorder('>')

        xarr = xr.DataArray(da.from_array(data[dataset_id.name]), dims=["y"])
        lat  = xr.DataArray(da.from_array(data['Latitude']), dims=["y"])
        lon  = xr.DataArray(da.from_array(data['Longitude']), dims=["y"])

        xarr['latitude']  = ('y',lat)
        xarr['longitude'] = ('y',lon)

        if xarr is None:

            dataset = None

        else:

            dataset = xarr

            dataset.attrs['units']         = dataset_info['units']
            dataset.attrs['wavelength']    = dataset_info['wavelength']
            dataset.attrs['standard_name'] = dataset_info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor']        = dataset_info['sensor']

        return dataset


class MSGAmvFileHandler(BaseFileHandler):
    """Reader for MSG AMV Final product in native format."""

    def __init__(self, filename, filename_info, filetype_info):
        super(MSGAmvFileHandler, self).__init__(filename, filename_info, filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat        = filename_info['subsat']
        self.rc_start      = filename_info['start_time']
        self.ssp_lon       = sub_sat_dict[filename_info['subsat']]

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start + timedelta(minutes=15)

    def _read_amv_header(self):
        """Read AMV product header."""
        hdr = np.fromfile(self.filename, SegProdHeaders.amvFinal_prd_hdr, 1)
        hdr = hdr.newbyteorder('>')
        return recarray2dict(hdr)

    def get_dataset(self, dataset_id, dataset_info):

        hdr_size = np.dtype(SegProdHeaders.amvFinal_prd_hdr).itemsize
        dtype    = np.dtype(SegProdDrecs.amvFinal)

        data = np.fromfile(self.filename, dtype=dtype, offset=hdr_size)
        data = data.newbyteorder('>')

        xarr = xr.DataArray(da.from_array(data[dataset_id.name]), dims=["y"])
        lat  = xr.DataArray(da.from_array(data['Latitude']), dims=["y"])
        lon  = xr.DataArray(da.from_array(data['Longitude']), dims=["y"])

        xarr['latitude']  = ('y',lat)
        xarr['longitude'] = ('y',lon)

        if xarr is None:

            dataset = None

        else:

            dataset = xarr

            dataset.attrs['units']         = dataset_info['units']
            dataset.attrs['wavelength']    = dataset_info['wavelength']
            dataset.attrs['standard_name'] = dataset_info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor']        = dataset_info['sensor']

        return dataset
