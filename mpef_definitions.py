import numpy as np

class GlobalTypes(object):

    # 8 bytes
    cds_time = [
                ('Day', np.uint16),
                ('MilliSecsOfDay', np.uint32),
                ('MicrosecsOfMillisecs', np.uint16)
                ]

    # 10 bytes
    cds_time_long = [
            ('Day', np.uint16),
            ('MilliSecsOfDay', np.uint32),
            ('MicrosecsOfMillisecs', np.uint16),
            ('NanosecsOfMicrosecs', np.uint16)
            ]

    # 6 bytes
    cds_time_short = [
                      ('Day', np.uint16),
                      ('MilliSecsOfDay', np.uint32)
                      ]

    # 4 bytes
    issue_revision = [
                      ('Issue', np.uint16),
                      ('Revision', np.uint16)
                      ]


class ImageNavigation(object):

    # 1024 bytes
    image_data_function = [
                           ('DataDefinitionString', (np.str, 1024))
                            ]

    aes_image_navigation = [
                        ('NoBitsPerPixel', np.int8),
                        ('Pad', (np.int8, 3)),
                        ('NoColumns', np.int16),
                        ('NoLines', np.int16),
                        ('ColScaling', np.int32),
                        ('LineScaling', np.int32),
                        ('ColOffset', np.int32),
                        ('LineOffset', np.int32),
                        ('NorthernLinePlanned', np.int16),
                        ('SouthernLinePlanned', np.int16),
                        ('ScanPeriod', np.int32),
                        ]
    # 48 bytes
    image_navigation = [
                        ('ProjectionName', (np.str, 32)),
                        ('ColScaling', np.int32),
                        ('LineScaling', np.int32),
                        ('ColOffset', np.int32),
                        ('LineOffset', np.int32),
                        ]

    # 16 bytes
    image_navigation_noproj = [
                               ('ColumnScalingFactor', np.int32),
                               ('LineScalingFactor', np.int32),
                               ('ColumnOffset', np.int32),
                               ('LineOffset', np.int32)
                               ]

                                # 8 bytes
    image_structure = [
                       ('NoBitsPerPixel', np.uint8),
                       ('Padding', 'S3'),
                       ('NoColumns', np.int16),
                       ('NoLines', np.int16)
                       ]


class MpefHeader(object):
    
    # 24 bytes
    images_used = [
                   ('Padding1', 'S2'),
                   ('ExpectedImage', GlobalTypes.cds_time_short),
                   ('ImageReceived', np.bool),
                   ('Padding2', 'S1'),
                   ('UsedImageStart_Day', np.uint16),
                   ('UsedImageStart_Millsec', np.uint32),
                   ('Padding3', 'S2'),
                   ('UsedImageEnd_Day', np.uint16),
                   ('UsedImageEndt_Millsec', np.uint32),
                   ]

    # 16 bytes
    mpef_time_generalized = [
                             ('mpef_time_generalized', (np.str, 16))
                             ]

    # 172 bytes
    mpef_product_header = [
                           ('MPEF_File_Id', np.int16),
                           ('MPEF_Header_Version', np.uint8),
                           ('ManualDissAuthRequest', np.bool),
                           ('ManualDisseminationAuth', np.bool),
                           ('DisseminationAuth', np.bool),
                           ('NominalTime', GlobalTypes.cds_time_short),
                           ('ProductQuality', np.uint8),
                           ('ProductCompleteness', np.uint8),
                           ('ProductTimeliness', np.uint8),
                           ('ProcessingInstanceId', np.int8),
                           ('ImagesUsed', images_used, (4,)),
                           ('BaseAlgorithmVersion',
                               GlobalTypes.issue_revision),
                           ('ProductAlgorithmVersion',
                               GlobalTypes.issue_revision),
                           ('InstanceServerName', 'S2'),
                           ('SpacecraftName', 'S2'),
                           ('Mission', 'S3'),
                           ('RectificationLongitude', 'S5'),
                           ('Encoding', 'S1'),
                           ('TerminationSpace', 'S1'),
                           ('EncodingVersion', np.uint16),
                           ('Channel', np.uint8),
                           ('Filler', 'S20'),
                           ('RepeatCycle', 'S15'),
                           ]


class ProdHeaders(object):    
    prod_hdr1  = [
            ('mpef_product_header',  MpefHeader.mpef_product_header),
            ('image_structure', ImageNavigation.aes_image_navigation),
        ]
        
    prod_hdr2 = [
                     ('mpef_product_header', MpefHeader.mpef_product_header),
                     ('image_structure', ImageNavigation.image_structure),
                     ('image_navigation_noproj',
                      ImageNavigation.image_navigation_noproj)
                ]
                
    prod_hdr3 = [
            ('mpef_product_header', MpefHeader.mpef_product_header),
            ('AccumStart', 'S16'),
            ('AccumEnd', 'S16'),
            ('image_structure',
                ImageNavigation.image_structure),
            ('image_navigation_noproj',
                ImageNavigation.image_navigation_noproj)
        ]

class ProdDrecs(object):
    aesInt = [
            ('AOT06', np.int16),
            ('AOT08', np.int16),
            ('AOT16', np.int16),
            ('AngstCoef', np.int16),
            ('Quality', np.int16)
            ]
            
    claInt = [
            ('SCE', np.uint8),
            ('EFF', np.uint8),
            ('CTP', np.uint16),
            ('CTT', np.uint16),
            ('Flags', np.uint8),
            ('SCEPerConf', np.uint8)
            ]
    
    clearSky = [
            ('NoAccums', np.uint8),
            ('Padding', 'S1'),
            ('SunZenMean', np.uint16),
            ('RelAziMean', np.uint16),
            ('aChanMean1', np.uint16),
            ('aChanMean2', np.uint16),
            ('aChanMean3', np.uint16),
            ('aChanMean4', np.uint16),
            ]
    
    ndvi = [
            ('Min', np.uint8),
            ('Max', np.uint8),
            ('Mean', np.uint8),
            ('Naccum', np.uint8)
            ]
            
    oca = [
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
            
    scenes = [
                ('SceneType', np.uint8),
                ('QualityFlag', np.uint8)
                ]                                    
