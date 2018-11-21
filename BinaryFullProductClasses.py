import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import dask.array as da

from satpy import CHUNK_SIZE
from pyresample import geometry

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.eum_base import recarray2dict
from .mpef_definitions import GlobalTypes, ImageNavigation,  MpefHeader
from .mpef_definitions import ProdHeaders, ProdDrecs
from .mpef_generic_functions import mpefGenericFuncs

logger = logging.getLogger('BinaryFullProductClasses')


sub_sat_dict = {"E0000": 0.0, "E0415": 41.5, "E0095": 9.5}


class MSGAesIntFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGAesIntFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}

        self.ssp_lon = sub_sat_dict[filename_info['subsat']]
        self.hdr_size = np.dtype(ProdHeaders.prod_hdr1).itemsize
        self.mda['number_of_lines'], self.mda['number_of_columns'] = \
            mpefGenericFuncs.read_header(ProdHeaders.prod_hdr1,
                                         filename)
        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on
           the actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int32),
                         ('LineRecord', ProdDrecs.aesInt,
                         (self.mda['number_of_lines'],))])

        dt = np.dtype([('DataRecord', drec, (1,))])
        dt = dt.newbyteorder('>')

        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(channel)]
        data = raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 32767)
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


class MSGClaIntFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGClaIntFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}

        self.mda['number_of_lines'], self.mda['number_of_columns'] = \
            mpefGenericFuncs.read_header(ProdHeaders.prod_hdr2, filename)
        self.ssp_lon = sub_sat_dict[filename_info['subsat']]
        self.hdr_size = np.dtype(ProdHeaders.prod_hdr2).itemsize

        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on the
           actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', ProdDrecs.claInt, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])
        dt = dt.newbyteorder('>')
        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(channel)].astype(np.uint16)

        raw1 = self.dask_array['DataRecord']['LineRecord'][:]

        data = raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        # xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0)
        xarr = xr.DataArray(data, dims=['y', 'x'])

        if xarr is None:
            dataset = None
        else:
            dataset = xarr

            if channel == 'CTT' or channel == 'CTP':
                dataset = dataset.where(dataset != 64537.)
            if channel == 'EFF' or channel == 'SCE':
                dataset = dataset.where(dataset != 0.)
            if channel == 'CTT':
                dataset += 170

            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset


class MSGClearSkyMapFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGClearSkyMapFileHandler, self).__init__(filename,
                                                        filename_info,
                                                        filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}

        self.ssp_lon = sub_sat_dict[filename_info['subsat']]
        self.hdr_size = np.dtype(ProdHeaders.prod_hdr3).itemsize
        self.mda['number_of_lines'], self.mda['number_of_columns'] = \
            mpefGenericFuncs.read_header(ProdHeaders.prod_hdr3, filename)

        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', ProdDrecs.clearSky,
                         (self.mda['number_of_columns'],))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        dt = dt.newbyteorder('>')

        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(dsid.name)]
        data = raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)
        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            if 'r' in dsid.name:
                dataset *= 0.1
            # dataset=dataset.where(dataset<50.)
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset


class MSGNdviFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGNdviFileHandler, self).__init__(filename,
                                                 filename_info,
                                                 filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}

        self.ssp_lon = sub_sat_dict[filename_info['subsat']]
        self.hdr_size = np.dtype(ProdHeaders.prod_hdr2).itemsize
        self.mda['number_of_lines'], self.mda['number_of_columns'] = \
            mpefGenericFuncs.read_header(ProdHeaders.prod_hdr2, filename)

        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on
           the actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int16),
                        ('Padding', np.int16),
                        ('LineRecord', ProdDrecs.ndvi, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(channel)]

        data = raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0)

        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            # Create new dataset where 255 values are ignored and interpreted
            # as nan's
            # dataset=dataset.where(dataset!=255,other=float('NaN'),drop=False)
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset


class MSGOcaFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGOcaFileHandler, self).__init__(filename,
                                                filename_info,
                                                filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}

        # OCA only contains MPH so this information has to be hardcoded for now
        self.hdr_size = np.dtype(MpefHeader.mpef_product_header).itemsize
        self.mda['number_of_lines'] = 3712
        self.mda['number_of_columns'] = 3712
        self.ssp_lon = sub_sat_dict[filename_info['subsat']]
        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on
           the actual available channels"""
        dt = np.dtype([('DataRecord', ProdDrecs.oca, (3712,))])
        dt = dt.newbyteorder('>')
        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):
        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['{}'.format(channel)]
        data = raw[:, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).astype(np.float32)
        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            if dsid.name == 'cot' or dsid.name == 'cot2':
                dataset = 10**dataset
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset


class MSGScenesFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGScenesFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}
        self.ssp_lon = sub_sat_dict[filename_info['subsat']]

        self.hdr_size = np.dtype(ProdHeaders.prod_hdr2).itemsize
        self.mda['number_of_lines'], self.mda['number_of_columns'] = \
            mpefGenericFuncs.read_header(ProdHeaders.prod_hdr2, filename)

        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', ProdDrecs.scenes,
                         (self.mda['number_of_columns'],))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name

        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['SceneType']
        data = raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data,
                            dims=['y', 'x']).where(data != 0).astype(np.float32)
        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            # dataset=dataset.where(dataset<50.)
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset


class MSGSstFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGSstFileHandler, self).__init__(filename,
                                                filename_info,
                                                filetype_info)

        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.mda = {}
        self.ssp_lon = sub_sat_dict[filename_info['subsat']]

        """Read the header info"""
        self.hdr_size = np.dtype(ProdHeaders.prod_hdr2).itemsize
        self.mda['number_of_lines'], self.mda['number_of_columns'] = \
            mpefGenericFuncs.read_header(ProdHeaders.prod_hdr2, filename)

        # Prepare dask-array
        self.dask_array = da.from_array(
            mpefGenericFuncs.get_memmap(self.filename,
                                        self._get_data_dtype(),
                                        self.mda['number_of_lines'],
                                        self.hdr_size),
                                        chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int16),
                        ('sst', np.int16, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])
        dt = dt.newbyteorder('>')

        return dt

    def get_area_def(self, dsid):
        return mpefGenericFuncs.get_area_def(dsid,
                                             self.mda['number_of_lines'],
                                             self.mda['number_of_columns'],
                                             self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['sst']

        data = raw[:, 0, :]
        shifted = np.left_shift(data, 5).astype(np.uint16)
        tmp = np.right_shift(shifted, 5)
        not_processed = np.where(tmp == 8)
        data = np.array(tmp.astype(np.float32))

        # data[not_processed] = np.nan

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x'])

        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            dataset *= 0.1
            dataset += 170.
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset
