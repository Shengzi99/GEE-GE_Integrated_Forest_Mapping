import numpy as np
import numba

import torch
import torch.nn as nn
import torch.utils.data as data

import osr
import os
import gdal
from tqdm import tqdm

import ee
import geemap

import GetMapTiles
import dataset


# def inference(model, data_loader, device=torch.device('cuda:0'), device_ids=(0, 1), comment="xxx", save_path="./CKPT/", pixelSize=(0.00026949, 0.00026949)):
#     """
#     使用model和data_loader加载ckpt进行预测，得到预测的图像类别和中心点偏置
#     """
#     save_path = save_path + "/" + comment
    
#     model.to(device)    
#     if os.path.exists(save_path + "/ckpt.pth"):
#         ckpt = torch.load(save_path + "/ckpt.pth", map_location=torch.device('cpu'))
#         model.load_state_dict(ckpt["model"])
#         model = nn.DataParallel(model, device_ids=device_ids)
#         model.eval()
        
#         all_pred = []
#         all_offset = None
#         with tqdm(data_loader, total=len(data_loader)) as t:
#             with torch.no_grad():
#                 for idx, img in enumerate(t):
#                     logits, CAM = model(img)
#                     pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
#                     offsets = calcOffset(pred, CAM, pixelSize=pixelSize)

#                     all_pred.extend(pred)
#                     if all_offset is None:
#                         all_offset = offsets
#                     else :
#                         all_offset = torch.cat([all_offset, offsets], dim=0)

#         return torch.tensor(all_pred), all_offset
#     else:
#         print("check point not found")

def inference(model, data_loader, device=torch.device('cuda:0'), pixelSize=(0.00026949, 0.00026949), desc=None):
    """
    使用model预测data_loader中的数据，得到预测的图像类别和中心点偏置
    """      
    model.eval()
    all_pred = []
    all_offset = None
    with tqdm(data_loader, total=len(data_loader), desc=desc) as t:
        with torch.no_grad():
            for idx, img in enumerate(t):
                img = img.to(device)
                logits, CAM = model(img)
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                offsets = calcOffset(pred, CAM, pixelSize=pixelSize)

                all_pred.extend(pred)
                if all_offset is None:
                    all_offset = offsets
                else :
                    all_offset = torch.cat([all_offset, offsets], dim=0)

    return torch.tensor(all_pred), all_offset

#----------------------------------------------------------------------------------------------------------------------------------------------
# 1. 相平面坐标和地理坐标互转相关函数
# ---------------------------------------------------------------------------------------------------------------------------------------------
def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]

# def imagexy2geo(dataset, row, col):
@numba.njit
def imagexy2geo(trans, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    # trans = dataset.GetGeoTransform()
    px = trans[0] + row * trans[1] + col * trans[2]
    py = trans[3] + row * trans[4] + col * trans[5]
    return px, py


#def geo2imagexy(dataset, x, y):
@numba.njit
def geo2imagexy(trans, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    # trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

#---------------------------------------------------------------------------------------------------------------------------------------------
# 2. 二值掩膜——>地理坐标List——>确定/不确定List
#---------------------------------------------------------------------------------------------------------------------------------------------

# def getLonLatListFromImage(imgPath):
#     if imgPath.split('.')[-1] not in ["tif", "tiff", "TIF", "TIFF"]:
#         raise Exception("input should be a GeoTiff image")
#     ds = gdal.Open(imgPath)
#     imgW, imgH = ds.RasterXSize, ds.RasterYSize
#     trans = ds.GetGeoTransform()
#     img = ds.ReadAsArray(0, 0, imgW, imgH)
#     lonlatList = []
#     for r in range(imgH):
#         for c in range(imgW):
#             geox, geoy = imagexy2geo(trans, c+0.5, r+0.5)
#             lon, lat = geox, geoy
#             # lon, lat = geo2lonlat(ds, geox, geoy)
#             sign = int(img[r, c])
#             lonlatList.append((lon, lat, sign))

#     return np.array(lonlatList)

def getLonLatListFromImage(imgPath):
    if imgPath.split('.')[-1] not in ["tif", "tiff", "TIF", "TIFF"]:
        raise Exception("input should be a GeoTiff image")
    ds = gdal.Open(imgPath)
    imgW, imgH = ds.RasterXSize, ds.RasterYSize
    trans = ds.GetGeoTransform()
    imgArray = ds.ReadAsArray(0, 0, imgW, imgH)
    return getLonLatListFromArray(imgArray, trans)


@numba.njit
def getLonLatListFromArray(imgArray, trans):
    imgW, imgH = imgArray.shape[1], imgArray.shape[0]
    lonlatList = []
    for r in range(imgH):
        for c in range(imgW):
            geox, geoy = imagexy2geo(trans, c+0.5, r+0.5)
            lon, lat = geox, geoy
            # lon, lat = geo2lonlat(ds, geox, geoy)
            sign = int(imgArray[r, c])
            lonlatList.append((lon, lat, sign))

    return np.array(lonlatList)


def getSP_SN_USsplit(llList_full):
    idx_unsure = np.where((llList_full[:, 2]==2) + (llList_full[:, 2]==3))[0]
    idx_sure_pos = np.where((llList_full[:, 2]==4) + (llList_full[:, 2]==5))[0]
    idx_sure_neg = np.where((llList_full[:, 2]==0) + (llList_full[:, 2]==1))[0]

    llList_unsure = llList_full[idx_unsure, ...].copy()
    llList_sure_pos = llList_full[idx_sure_pos, ...].copy()
    llList_sure_neg = llList_full[idx_sure_neg, ...].copy()
    
    return llList_sure_pos, llList_sure_neg, llList_unsure
    
#---------------------------------------------------------------------------------------------------------------------------------------------
# 3. 由CAM获取中心点偏置
#---------------------------------------------------------------------------------------------------------------------------------------------

def calcOffset(prediction, CAM, pixelSize=None):
    """
    通过CAM获取中心点偏置
    Params:
        prediction - CNN预测的类别结果，形状[N]
        CAM - CNN输出的Class Activation Map，形状[N, C, H, W]
        pixelSize - 一个CAM像元在lon和lat方向的大小，单位为度。如果输入为None则返回offsets值单位为像元个数, tuple(longitude_size, latitude_size)
    Return:
        offsets - 计算得到的每一张图片在longitude和latitude方向的offset，形状[N, 2]
    """

    # 相平面坐标轴y和纬度坐标轴是相反的
    pixelSize = (pixelSize[0], -pixelSize[1])
    idx_pad = torch.arange(0, CAM.shape[0]*CAM.shape[1], CAM.shape[1], device=CAM.device)
    idx = idx_pad + prediction

    CAM_selected = CAM.view(-1, CAM.shape[2], CAM.shape[3]).index_select(0, idx)
    CAM_argmax = torch.argmax(CAM_selected.view(CAM_selected.shape[0], -1), dim=1)
    
    if pixelSize is None:
        offsets = torch.stack([CAM_argmax % CAM.shape[3] - (CAM.shape[3] // 2), CAM_argmax // CAM.shape[3] - (CAM.shape[2] // 2)], dim=-1)
    else:
        offsets = torch.stack([(CAM_argmax % CAM.shape[3] - (CAM.shape[3] // 2)) * pixelSize[0], (CAM_argmax // CAM.shape[3] - (CAM.shape[2] // 2)) * pixelSize[1]], dim=-1)
    
    return offsets

#---------------------------------------------------------------------------------------------------------------------------------------------
# 4. 下载：0.1d高分、0.1d融合、0.5d融合
#---------------------------------------------------------------------------------------------------------------------------------------------

def download_ffge(savePath, feature05, feature01, forest_fuse):
    ID_5d = feature05.getNumber("ID_5d").getInfo()
    ID_05d = feature01.getNumber("ID_05d").getInfo()
    ID_01d = feature01.getNumber("ID_01d").getInfo()
    
    assert ID_05d == feature05.getNumber("ID_05d").getInfo(), "0.1d feature should match with 0.5d feature"
    assert ID_5d == feature01.getNumber("ID_5d").getInfo(), "0.1d feature should match with 0.5d feature"

    if not os.path.exists(savePath + "grid%d" % ID_5d):
        os.mkdir(savePath + "grid%d" % ID_5d)

    path_ge = savePath + "grid%d/ge_%d_%d_%d.tif" % (ID_5d, ID_5d, ID_05d, ID_01d)
    path_ff01 = savePath + "grid%d/ff01_%d_%d_%d.tif" % (ID_5d, ID_5d, ID_05d, ID_01d)
    path_ff05 = savePath + "grid%d/ff05_%d_%d.tif" % (ID_5d, ID_5d, ID_05d)

    if not os.path.exists(path_ge):
        xmin01, ymin01 = feature01.getNumber("xmin").getInfo(), feature01.getNumber("ymin").getInfo()
        padding = 0.002
        GetMapTiles.getpic_tif(xmin01-padding, ymin01+0.1+padding, xmin01+0.1+padding, ymin01-padding, 17, source='google', out_filename=path_ge, style='s')
    else:
        print("%s already exits" % path_ge.split('/')[-1])
    if not os.path.exists(path_ff01):
        geemap.ee_export_image(forest_fuse, path_ff01, scale=30, region=feature01.geometry())
    else:
        print("%s already exits" % path_ff01.split('/')[-1])
    if not os.path.exists(path_ff05):
        geemap.ee_export_image(forest_fuse, path_ff05, scale=30, region=feature05.geometry())
    else:
        print("%s already exits" % path_ff05.split('/')[-1])

def checkExists_ffge(dataPath, ID_5d, IDlist_05d, IDlist_01d):
    dataPathCur = dataPath + "/grid%d" % ID_5d
    assert len(IDlist_05d) == len(IDlist_01d), "List length of 05d and 01d should be same"
    list_len = len(IDlist_05d)

    ge_exists, ff01_exists, ff05_exists = 0, 0, 0
    for idx in range(len(IDlist_01d)):
        ge_exists += int(os.path.exists(dataPathCur + "/ge_%d_%d_%d.tif" % (ID_5d, IDlist_05d[idx], IDlist_01d[idx])))
        ff01_exists += int(os.path.exists(dataPathCur + "/ff01_%d_%d_%d.tif" % (ID_5d, IDlist_05d[idx], IDlist_01d[idx])))
        ff05_exists += int(os.path.exists(dataPathCur + "/ff05_%d_%d.tif" % (ID_5d, IDlist_05d[idx])))   
    ge_ready, ff01_ready, ff05_ready = (ge_exists == list_len), (ff01_exists == list_len), (ff05_exists == list_len)

    if ge_ready and ff01_ready and ff05_ready:
        print("data ready!")
        return True
    else:
        print("missing data:%s%s%s, waiting for download..." % ((" GE" if not ge_ready else ""), (" FF01" if not ff01_ready else ""), (" FF05" if not ff05_ready else "")))
        return False

#---------------------------------------------------------------------------------------------------------------------------------------------
# 5. 从确定区域（直接从融合产品提取）和不确定区域（CNN预测）提取随机森林样本点
#---------------------------------------------------------------------------------------------------------------------------------------------
def getRFSampleList(llList_sure_pos, llList_sure_neg, llList_unsure, CNNmodel, gePath, data_loader=None, device=torch.device('cuda:0'), desc=None):

    dataLoader = dataset.getLonLatDataLoader(llList_unsure[:, 0:2], local=True, imgSavePath=gePath, BSize=32, nWorkers=0, pinMem=True, APIKEY=None)
    if data_loader is not None:
        dataLoader = data_loader
    pred, offset = inference(CNNmodel, data_loader=dataLoader, pixelSize=(0.00026949, 0.00026949), device=device, desc=desc)
    predforest = pred.cpu().numpy().astype(np.int)

    # get all sample as 0-1 list
    llList_forest_sure_pos = llList_sure_pos.copy()
    llList_forest_sure_neg = llList_sure_neg.copy()
    llList_forest_unsure = llList_unsure.copy()

    llList_forest_sure_pos[:, 2] = (llList_sure_pos[:, 2]>=4).astype(np.int)
    llList_forest_sure_neg[:, 2] = (llList_sure_neg[:, 2]>=4).astype(np.int)
    llList_forest_unsure[:, 2] = predforest

    idx_labeled = np.where(llList_forest_unsure[:, 2]!=4)[0]
    llList_forest_unsure = llList_forest_unsure[idx_labeled, ...]
    llList_forest_unsure[:, 2] = (llList_forest_unsure[:, 2].copy()==0).astype(np.int)


    # apply offset to llList_forest_unsure
    offset = offset[idx_labeled, ...]
    llList_forest_unsure_offseted = llList_forest_unsure.copy()
    llList_forest_unsure_offseted[:, [0, 1]] += offset.cpu().numpy()

    llList_forest_sure_pos = llList_forest_sure_pos.tolist()
    llList_forest_sure_neg = llList_forest_sure_neg.tolist()

    llList_forest_unsure = llList_forest_unsure.tolist()
    llList_forest_unsure_offseted = llList_forest_unsure_offseted.tolist()

    return llList_forest_sure_pos + llList_forest_sure_neg, llList_forest_unsure, llList_forest_unsure_offseted


def getRFSampleFC(image, llList_forest_sample):

    CrdClsList = ee.List(llList_forest_sample)
    mapped = CrdClsList.map(lambda x : ee.Feature(ee.Algorithms.GeometryConstructors.Point(ee.List(x).slice(0, 2)), {"forest":ee.List(x).getNumber(2).int()}))
    fc_points = ee.FeatureCollection(mapped)
    sample_train = image.sampleRegions(collection=fc_points, geometries=True, scale=30)
    
    return sample_train
