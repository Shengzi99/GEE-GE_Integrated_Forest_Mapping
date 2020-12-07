import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import osr
import os
import gdal
from tqdm import tqdm


def inference(model, data_loader, device=torch.device('cuda:0'), device_ids=(0, 1), comment="xxx", save_path="./CKPT/", pixelSize=(0.00026949, 0.00026949)):
    """
    使用model和data_loader加载ckpt进行预测，得到预测的图像类别和中心点偏置
    """
    save_path = save_path + "/" + comment
    
    model.to(device)    
    if os.path.exists(save_path + "/ckpt.pth"):
        ckpt = torch.load(save_path + "/ckpt.pth", map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])
        model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()
        
        all_pred = []
        all_offset = None
        with tqdm(data_loader, total=len(data_loader)) as t:
            with torch.no_grad():
                for idx, img in enumerate(t):
                    logits, CAM = model(img)
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    offsets = calcOffset(pred, CAM, pixelSize=pixelSize)

                    all_pred.extend(pred)
                    if all_offset is None:
                        all_offset = offsets
                    else :
                        all_offset = torch.cat([all_offset, offsets], dim=0)

        return torch.tensor(all_pred), all_offset
    else:
        print("check point not found")

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

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + row * trans[1] + col * trans[2]
    py = trans[3] + row * trans[4] + col * trans[5]
    return px, py


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

#---------------------------------------------------------------------------------------------------------------------------------------------
# 2. 二值掩膜——>地理坐标List
#---------------------------------------------------------------------------------------------------------------------------------------------

def getLonLatListFromImage(imgPath):
    if imgPath.split('.')[-1] not in ["tif", "tiff", "TIF", "TIFF"]:
        raise Exception("input should be a GeoTiff image")
    ds = gdal.Open(imgPath)
    imgW, imgH = ds.RasterXSize, ds.RasterYSize
    img = ds.ReadAsArray(0, 0, imgW, imgH)
    lonlatList = []
    for r in range(imgH):
        for c in range(imgW):
            geox, geoy = imagexy2geo(ds, c+0.5, r+0.5)
            lon, lat = geo2lonlat(ds, geox, geoy)
            sign = int(img[r, c])
            lonlatList.append((lon, lat, sign))

    return np.array(lonlatList)

#---------------------------------------------------------------------------------------------------------------------------------------------
# 3. 由CAM获取中心点偏执。及将偏置应用于LonLatList
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


if __name__ == "__main__":
    pred = torch.randint(0, 5, size=(1024, ))
    CAM = torch.randint(0, 10, size=(1024, 5, 5, 5))
    CAM = CAM/CAM.max().float()

    print(pred.shape, CAM.shape)
    offsets = calcOffset(pred, CAM)
    print(offsets)