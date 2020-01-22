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

    # 32 bytes
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
        ('BaseAlgorithmVersion', GlobalTypes.issue_revision),
        ('ProductAlgorithmVersion', GlobalTypes.issue_revision),
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

    prod_hdr1 = [
        ('mpef_product_header',  MpefHeader.mpef_product_header),
        ('image_structure', ImageNavigation.aes_image_navigation),
        ]

    prod_hdr2 = [
        ('mpef_product_header', MpefHeader.mpef_product_header),
        ('image_structure', ImageNavigation.image_structure),
        ('image_navigation_noproj', ImageNavigation.image_navigation_noproj)
        ]

    prod_hdr3 = [
        ('mpef_product_header', MpefHeader.mpef_product_header),
        ('AccumStart', 'S16'),
        ('AccumEnd', 'S16'),
        ('image_structure', ImageNavigation.image_structure),
        ('image_navigation_noproj', ImageNavigation.image_navigation_noproj)
        ]


class SegProdHeaders(object):

    prod_hdr1 = [
        ('mpef_product_header', MpefHeader.mpef_product_header),
        ('SegmentWidth', np.int16),
        ('SegmentHeight', np.int16),
        ('NoSegsInProduct', np.int32)
        ]

    amvIntm_prd_hdr = [
        ('mpef_product_header', MpefHeader.mpef_product_header),
        ('ChannelID', np.uint8),
        ('ProcSegWidth', np.uint8),
        ('ProcSegHeight', np.uint8),
        ('CloudTargetWidth', np.uint8),
        ('CloudTargetHeight', np.uint8),
        ('ClearSkyTargetWidth', np.uint8),
        ('ClearSkyTargetHeight', np.uint8),
        ('CloudSearchWidth', np.uint8),
        ('CloudSearchHeight', np.uint8),
        ('ClearSearchWidth', np.uint8),
        ('ClearSearchHeight', np.uint8),
        ('Pad', np.uint8),
        ('NumAMV', np.int32)
        ]

    cds_time = [
        ('Day', np.int32),
        ('Second', np.int32)
        ]

    time_hour = [
        ('Hour', np.int16),
        ('Minute', np.int8),
        ('Second', np.int8)
        ]

    amvFinal_prd_hdr = [
        ('mpef_product_header', MpefHeader.mpef_product_header),
        ('ChannelID', np.uint8),
        ('Pad1', (np.uint8, 3)),
        ('Frequency', np.float32),
        ('BandWith', np.float32),
        ('ProcSegWidth', np.uint8),
        ('ProcSegHeight', np.uint8),
        ('Pad2', (np.uint8, 2)),
        ('SegWidth', np.uint32),
        ('SegHeight', np.uint32),
        ('CorrMethod', np.uint32),
        ('NumAMV', np.int32),
        ('NumPassAQC', np.int32),
        ('NumCycles', np.uint8),
        ('Pad3', (np.uint8, 3)),
        ('RepeatCycleTimes', cds_time, (10,)),
        ('ProdYear', np.uint16),
        ('ProdMonth', np.uint16),
        ('ProdDay', np.uint16),
        ('ProdHour', np.uint16),
        ('StartTime', time_hour, (3,)),
        ('EndTime', time_hour, (3,)),
        ('Padding7', (np.uint8, 48)),
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

    fire = [
        ('afm', np.uint8),
        ('pad', np.uint8),
        ('prob', np.uint16),
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


class SegProdDrecs(object):

    clarec1 = [
        ('CloudTypes', np.uint8),
        ('CloudAmounts', np.uint8),
        ('CloudPhases', np.uint8),
        ('Padding',  np.uint8),
        ('CloudTopTemps', np.float32),
        ('CloudTopHeights', np.float32)
        ]

    clarec2 = [
        ('LevCloudAmount', np.uint8),
        ('NoCloudTypes', np.uint8),
        ('Padding',  np.uint16),
        ('CloudStats', clarec1, (3,))
        ]

    clarec3 = [
        ('SegmentRow', np.int32),
        ('SegmentCol', np.int32),
        ('Latitude', np.float32),
        ('Longitude', np.float32),
        ('TotEffCloudAmount', np.uint8),
        ('QualityFlag', np.uint8),
        ('Padding',  np.uint16),
        ('Data', clarec2, (3,))
        ]

    gii = [
        ('Kindex', np.float32),
        ('KOindex', np.float32),
        ('LiftingIndex', np.float32),
        ('MaxBuoyancy', np.float32),
        ('TotalPrecWat', np.float32),
        ('PercentClear', np.float32),
        ('Row', np.float32),
        ('Column', np.float32),
        ('Latitude', np.float32),
        ('Longitude', np.float32),
        ('SatZenithAngle', np.float32),
        ('Layer1PrecWater', np.float32),
        ('Layer2PrecWater', np.float32),
        ('Layer3PrecWater', np.float32)
        ]

    heightAss = [
        ('Pressure', np.float32),
        ('PressureSd', np.float32),
        ('TempUncorr', np.float32),
        ('TempCorr', np.float32),
        ('TempCorrSd', np.float32),
        ('Confidence', np.float32),
        ('PixelFraction', np.float32)
        ]

    amvIntm = [
        ('TargetID', np.int32),
        ('Latitude', np.float32),
        ('Longitude', np.float32),
        ('Speed', np.float32),
        ('Direction', np.float32),
        ('TempUncorr', np.float32),
        ('HeightUncorr', np.float32),
        ('CorrMethod', np.float32),
        ('Temperature', np.float32),
        ('Height', np.float32),
        ('MaxCorr', np.float32),
        ('Row', np.uint8),
        ('Col', np.uint8),
        ('QI', np.uint8),
        ('HeightError', np.uint8),
        ('TargetType', np.uint8),
        ('ImageEnhancement', np.uint8),
        ('NbrColdPixels', np.int16),
        ('QualResults', (np.uint8, 12)),
        ('LandFrac', np.int32),
        ('HeightAss', heightAss, (19,))
        ]

    windSummary = [
        ('Direction', np.float32),
        ('Speed', np.float32),
        ('WindU', np.float32),
        ('WindV', np.float32)
        ]

    heightSummary = [
        ('Pressure', np.float32),
        ('PressureSd', np.float32),
        ('Temperature', np.float32),
        ('TemperatureSd', np.float32)
        ]

    bufrSummary = [
        ('BUFRCode', np.float32),
        ('Pressure', np.float32),
        ('Temperature', np.float32)
        ]

    amvFinal = [
        ('Latitude', np.float32),
        ('Longitude', np.float32),
        ('Speed', np.float32),
        ('Direction', np.float32),
        ('WindU', np.float32),
        ('WindV', np.float32),
        ('Temperature', np.float32),
        ('Height', np.float32),
        ('QI', np.uint8),
        ('QIExcludingFcst', np.uint8),
        ('HeightError', np.uint8),
        ('QualResults', (np.uint8, 11)),
        ('ChannelID', np.uint8),
        ('RFFQuality', np.uint8),
        ('SatZenithAngle', np.float32),
        ('TargetType', np.uint8),
        ('HeightConsist', np.uint8),
        ('Pad1', (np.uint8, 2)),
        ('WindMethod', np.int32),
        ('LandFrac', np.int32),
        ('LandSeaFlag', np.uint8),
        ('Pad2', (np.uint8, 3)),
        ('WindSum', windSummary, (3,)),
        ('HeightSum', heightSummary, (19,)),
        ('BUFRSum', bufrSummary, (10,)),
        ('PressureSd', np.float32),
        ('PressureSdOCA', np.float32)
        ]
