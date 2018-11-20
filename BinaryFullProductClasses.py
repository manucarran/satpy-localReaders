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
from .mpef_generic_functions import mpefGenericFuncs

logger = logging.getLogger('BinaryFullProductClasses')


class MSGAesIntFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGAesIntFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time

    @property
    def end_time(self):
        end_time = self.rc_start+timedelta(minutes=15)
        return end_time

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        self.parameters = {
            'aot06': 'AOT06',
            'aot08': 'AOT08',
            'aot16': 'AOT16',
            'ang': 'AngstCoef',
            'qlty': 'Quality',
        }

        lrec = [
            ('AOT06', np.int16),
            ('AOT08', np.int16),
            ('AOT16', np.int16),
            ('AngstCoef', np.int16),
            ('Quality', np.int16)
            ]

        drec = np.dtype([('ImageLineNo', np.int32),
                         ('LineRecord', lrec, (self.mda['number_of_lines'],))])

        dt = np.dtype([('DataRecord', drec, (1,))])
        dt = dt.newbyteorder('>')

        return dt

    def _read_header(self):
        """Read the header info"""
        _header_record = [
            ('mpef_product_header', MpefHeader.mpef_product_header),
            ('image_structure', ImageNavigation.aes_image_navigation),
        ]

        self.header = np.fromfile(self.filename,
                                  dtype=_header_record, count=1)

        self.hdr_size = np.dtype(_header_record).itemsize

        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
            self.ssp_lon = 0.0

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] = self.header['image_structure']['NoColumns'][0]

        # Check the calculated row,column dimensions against the header information:
        # ncols = self.mda['number_of_columns']

    def get_area_def(self, dsid):
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid, nlines, ncols, self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = self.parameters[dsid.name]

        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(channel)]
        data = raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        # xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)
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
        super(MSGClaIntFileHandler, self).__init__(filename, filename_info, filetype_info)
        print ('filetype info', filename_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        print ('start time', self.rc_start)
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))
            
    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time

    @property
    def end_time(self):
        end_time = self.rc_start+timedelta(minutes=15)
        return end_time

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        lrec = [
            ('SCE', np.uint8),
            ('EFF', np.uint8),
            ('CTP', np.uint16),
            ('CTT', np.uint16),
            ('Flags', np.uint8),
            ('SCEPerConf', np.uint8)
            ]

        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', lrec, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])
        dt = dt.newbyteorder('>')
        return dt

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

        self.hdr_size = np.dtype(header_record).itemsize

        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
            self.ssp_lon = 0.0

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] = self.header['image_structure']['NoColumns'][0]

        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']

    def get_area_def(self, dsid):
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid, nlines, ncols, self.ssp_lon)

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
        super(MSGClearSkyMapFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time

    @property
    def end_time(self):
        end_time = self.rc_start+timedelta(minutes=15)
        return end_time

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        self.parameters = {
            'azi': 'RelAziMean',
            'r06': 'aChanMean1',
            'r08': 'aChanMean2',
            'r16': 'aChanMean3',
            'sza': 'SunZenMean'
        }

        lrec = [
            ('NoAccums', np.uint8),
            ('Padding', 'S1'),
            ('SunZenMean', np.uint16),
            ('RelAziMean', np.uint16),
            ('aChanMean1', np.uint16),
            ('aChanMean2', np.uint16),
            ('aChanMean3', np.uint16),
            ('aChanMean4', np.uint16),
            ]

        drec = np.dtype([('ImageLineNo', np.int16),
                         ('LineRecord', lrec, (self.mda['number_of_columns'],))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        dt = dt.newbyteorder('>')

        return dt

    def _read_header(self):
        """Read the header info"""
        header_record = [
            ('mpef_product_header', MpefHeader.mpef_product_header),
            ('AccumStart', 'S16'),
            ('AccumEnd', 'S16'),
            ('image_structure',
                ImageNavigation.image_structure),
            ('image_navigation_noproj',
                ImageNavigation.image_navigation_noproj)
        ]

        self.header = np.fromfile(self.filename,
                                  dtype=header_record, count=1)

        self.hdr_size = np.dtype(header_record).itemsize

        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
            self.ssp_lon = 0.0

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] = self.header['image_structure']['NoColumns'][0]

        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']
    
    def get_area_def(self, dsid):
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid, nlines, ncols, self.ssp_lon)
        
    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = self.parameters[dsid.name]

        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(channel)]
        data=raw[:, 0, :]

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
        super(MSGNdviFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))
            
    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time

    @property
    def end_time(self):
        end_time = self.rc_start+timedelta(minutes=15)
        return end_time

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        lrec = [
            ('Min', np.uint8),
            ('Max', np.uint8),
            ('Mean', np.uint8),
            ('Naccum', np.uint8)
            ]

        drec = np.dtype([('ImageLineNo', np.int16),
                        ('Padding', np.int16),
                        ('LineRecord', lrec, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])

        return dt
        
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

        self.hdr_size = np.dtype(header_record).itemsize

        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
            self.ssp_lon = 0.0

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] = self.header['image_structure']['NoColumns'][0]

        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']

    def get_area_def(self, dsid):
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid, nlines, ncols, self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['{}'.format(channel)]

        data=raw[:, 0, :]

        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0)

        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            # Create new dataset where 255 values are ignored and interpreted
            # as nan's
            #dataset=dataset.where(dataset!=255,other=float('NaN'),drop=False)
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset

class MSGOcaFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGOcaFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                    self._get_data_dtype(),
                                                                    self.mda['number_of_lines'],
                                                                    self.hdr_size), chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time

    @property
    def end_time(self):
        end_time = self.rc_start+timedelta(minutes=15)
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
        
    def _read_header(self):
        """Read the header info"""
        header_record = [
                     ('mpef_product_header', MpefHeader.mpef_product_header),
                     ]

        self.header = np.fromfile(self.filename,
                           dtype=header_record, count=1)
        
        self.hdr_size = np.dtype(header_record).itemsize 
            
        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
             self.ssp_lon = 0.0                

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = 3712
        self.mda['number_of_columns']= 3712
        
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
        data = raw[:, :]
        
        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).astype(np.float32)
        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            if dsid.name == 'cot' or dsid.name == 'cot2':
                dataset=10**dataset
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'
        
        return dataset


class MSGScenesFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGScenesFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        print ('In Binary Full')
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))
        
    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time
        
    @property
    def end_time(self):
        end_time = self.rc_start+timedelta(minutes=15)
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
            
        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
            self.ssp_lon = 0.0                

        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] =  self.header['image_structure']['NoColumns'][0]
        
        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']
    
    def get_area_def(self, dsid):        
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid,nlines,ncols,self.ssp_lon)
        
    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name

        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['LineRecord']['SceneType']
        data=raw[:, 0, :]
        
        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)
        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            #dataset=dataset.where(dataset<50.) 
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset
        
class MSGSstFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(MSGSstFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name=filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.header = {}
        self.mda = {}
        self._read_header()
        # Prepare dask-array
        self.dask_array = da.from_array(mpefGenericFuncs.get_memmap(self.filename,
                                                                   self._get_data_dtype(),
                                                                   self.mda['number_of_lines'],
                                                                   self.hdr_size), chunks=(CHUNK_SIZE,))
        
    @property
    def start_time(self):
        start_time = self.rc_start
        return start_time
        
    @property
    def end_time(self):
        end_time=self.rc_start+timedelta(minutes=15)
        return end_time
	    
    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""
        drec = np.dtype([('ImageLineNo', np.int16),
                        ('sst', np.int16, (3712,))])

        dt = np.dtype([('DataRecord', drec, (1,))])
        dt = dt.newbyteorder('>')
        
        return dt
        
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
        
        if self.subsat == 'E0000':
            self.ssp_lon = 0.0
        elif self.subsat == 'E0415':
            self.ssp_lon = 41.5
        elif self.subsat == 'E0095':
            self.ssp_lon = 9.5
        else:
             self.ssp_lon = 0.0                
        
        self.header = self.header.newbyteorder('>')
        self.mda['number_of_lines'] = self.header['image_structure']['NoLines'][0]
        self.mda['number_of_columns'] =  self.header['image_structure']['NoColumns'][0]
        
        # Check the calculated row,column dimensions against the header information:
        ncols = self.mda['number_of_columns']
        
    def get_area_def(self, dsid):
        
        nlines = self.mda['number_of_lines']
        ncols = self.mda['number_of_columns']
        return mpefGenericFuncs.get_area_def(dsid,nlines,ncols,self.ssp_lon)

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        raw = self.dask_array['DataRecord']['sst']
        
        data=raw[:, 0, :]
        shifted = np.left_shift(data, 5).astype(np.uint16)
        tmp = np.right_shift(shifted, 5)
        not_processed = np.where(tmp == 8)
        data = np.array(tmp.astype(np.float32))
        
        #data[not_processed] = np.nan
        
        data = da.flipud(da.fliplr((data.reshape(shape))))
        xarr = xr.DataArray(data, dims=['y', 'x'])

        if xarr is None:
            dataset = None
        else:
            dataset = xarr
            dataset*=0.1    
            dataset+=170.
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'
        
        return dataset

