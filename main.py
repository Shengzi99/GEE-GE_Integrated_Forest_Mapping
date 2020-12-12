# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 1. Authenticate & Import

# %%
import ee
ee.Initialize()

import geemap


# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

import gdal
import utils
import common
import dataset
import GetMapTiles
from models.ResNet import ResNet101

# %% [markdown]
# # 2. Import Earth Engine Data

# %%
if __name__ == "__main__":
    # Boundary and Grid-----------------------------------------------------------------------------------------------------
    worldBoundary = ee.FeatureCollection("users/liuph/shape/WorldBoundary")
    ChinaBoundary = ee.FeatureCollection("users/410093033/China")
    WorldGrid5d = ee.FeatureCollection("users/liuph/shape/WorldGrid5dC5")

    # Landcover products----------------------------------------------------------------------------------------------------

    gong = ee.ImageCollection('users/wangyue/Gong2017Glc30') # gong's data
    forest_gong = ee.ImageCollection(gong).qualityMosaic("b1").expression("b(0)==2?1:0").rename("Forestgong")

    ygs = ee.ImageCollection('users/sunly3456/Forest2018ImageCollection') # ygs's data
    forest_ygs =  ee.ImageCollection(ygs).qualityMosaic("b1").expression("b(0)==1?1:0").rename("Forestygs")

    liu = ee.ImageCollection('users/wangyue/Glc2020Fcs30').select('b1') # liu's data
    forest_liu = ee.ImageCollection(liu).qualityMosaic("b1").expression("b(0)>=50 && b(0)<=90?1:0").rename("Forestliu")

    lc = ee.ImageCollection('users/sunly3456/GLC2020') # Chen's data
    forest_chen = ee.ImageCollection(lc).qualityMosaic("b1").expression("b(0)==20?1:0").rename("Forestchen")
        
    hansen = ee.Image('UMD/hansen/global_forest_change_2019_v1_7') # Hansen's data
    start2000 = ee.Image(hansen).select('treecover2000').expression("b(0)>30&&b(0)<=100?1:0").rename("start00")
    loss00_19 = ee.Image(hansen).expression("b(3)>1&&b(3)<=19?1:0").rename("loss00_19")
    gain00_12 = ee.Image(hansen).expression("b(2)==1?1:0").rename("gain00_12")
    forest_hs = start2000.add(gain00_12).subtract(loss00_19).expression("b(0)>0?1:0").rename("Foresths")

    # Fusion of landcover products------------------------------------------------------------------------------------------
    forest_fuse = forest_gong.add(forest_ygs).add(forest_liu).add(forest_chen).add(forest_hs).rename("ForestFuse")
    # forest_fuse = ee.ImageCollection('users/sysushiqian/forestFuse2020').min().rename("forest_fuse")
    forest23 = forest_fuse.expression('b(0)>=2 && b(0)<=3')

    # Landsat8, cloud mask, median; add NDVI,NDWI,slop ---------------------------------------------------------------------
    Landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
    selVar = ee.List(['B2', 'B3', 'B4', 'B5', 'B6', 'B7','pixel_qa'])
    LC = Landsat.filter(ee.Filter.calendarRange(2020, 2020, 'year')).select(selVar).map(utils.maskL8sr)
    selVar = ee.List(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    LC = LC.select(selVar).median()

    ndvi = LC.normalizedDifference(['B5', 'B4']).rename('NDVI')
    ndwi = LC.normalizedDifference(['B3', 'B5']).rename('NDWI')
    DEM = ee.Image("MERIT/DEM/v1_0_3")
    terrain = ee.Algorithms.Terrain(DEM)
    slope = terrain.select('slope')
    stratified_classes = ndvi.expression('(b(0)/0.2)').int().rename('STRATIFIED_CLASSES')

    # Composite Image ------------------------------------------------------------------------------------------------------
    LC_STN = LC.addBands(ndvi).addBands(ndwi).addBands(slope).addBands(stratified_classes).float()
    selVar1 = ee.List(['B2', 'B3', 'B4', 'B5', 'B6', 'B7','NDVI','NDWI','slope'])

    # %% [markdown]
    # # 3. Prepare CNN model & data path

    # %%
    dataPath = r"E:\SZT\Data\ff0105_ge01"

    ckptPath = r"./CKPT/ResNet101_GE17_1206/ckpt.pth"

    model = ResNet101(in_ch=3, n_classes=5)
    model.to(torch.device('cuda:0'))    
    assert os.path.exists(ckptPath), "ckpt dosen't exists"
    ckpt = torch.load(ckptPath, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model"])
    model = nn.DataParallel(model, device_ids=(0, ))

    # %% [markdown]
    # # 4. Operate on selected 5d grid

    # %%
    ID_grid5d = 1038

    dataPath_curGrid = dataPath + "./grid%d" % ID_grid5d

    selected5d = WorldGrid5d.filterMetadata("ID", "equals", ID_grid5d).first()
    grid05d, grid01d = utils.get0501Grid(selected5d, forest23)

    idList_05d = grid05d.reduceColumns(ee.Reducer.toList(), ["ID_05d"]).get("list").getInfo()


    # %%
    # for i in idList_05d[95:]:
    #     cur_grid = grid01d.filterMetadata("ID_05d", "equals", i).first()
    #     cur_xmin, cur_ymin, ID_01d = cur_grid.getNumber("xmin"), cur_grid.getNumber("ymin"), cur_grid.getNumber("ID_01d").getInfo()
    #     padded_feature = ee.FeatureCollection(ee.Algorithms.GeometryConstructors.Rectangle(ee.List([cur_xmin.subtract(0.002), cur_ymin.subtract(0.002), cur_xmin.add(0.102), cur_ymin.add(0.102)])))
    #     geemap.ee_export_vector(padded_feature, "./shape/ge_%d_%d_%d.kml"%(ID_grid5d, i, ID_01d))


    # %%
    # for i in idList_05d:
    #     common.download_ffge(dataPath_curGrid, grid05d.filterMetadata("ID_05d", "equals", i).first(), grid01d.filterMetadata("ID_05d", "equals", i).first(), forest_fuse)


    # %%
    llListFS_list = []
    llListFU_list = []
    llListFU_offseted_list = []

    time_start = time.time()
    for idx, ID05d in enumerate(idList_05d):
        print("started from %s, passsed %d sec, processing %d/100 0.5d grid: " % (time.ctime(time_start), time.time()-time_start, idx+1))
        ID01d = grid01d.filterMetadata("ID_05d", "equals", ID05d).first().getNumber("ID_01d").getInfo()
        # get full llList
        llList05_full = common.getLonLatListFromImage(dataPath_curGrid + "/ff05_%d_%d.tif" % (ID_grid5d, ID05d))
        llList01_full = common.getLonLatListFromImage(dataPath_curGrid + "/ff01_%d_%d_%d.tif" % (ID_grid5d, ID05d, ID01d))
        llList05_sure_pos, llList05_sure_neg, llList05_unsure = common.getSP_SN_USsplit(llList05_full)
        llList01_sure_pos, llList01_sure_neg, llList01_unsure = common.getSP_SN_USsplit(llList01_full)
        llList_sure_pos, llList_sure_neg, llList_unsure = llList05_sure_pos, llList05_sure_neg, llList01_unsure

        # sample from llList
        sampleNum_sure = min([4000, llList_sure_pos.shape[0], llList_sure_neg.shape[0]])
        sampleNum_unsure = min(4000, llList_unsure.shape[0])

        np.random.seed(0)
        sp_idx = np.random.choice(np.arange(llList_sure_pos.shape[0]), size=sampleNum_sure, replace=False, p=None)
        sn_idx = np.random.choice(np.arange(llList_sure_neg.shape[0]), size=sampleNum_sure, replace=False, p=None)
        us_idx = np.random.choice(np.arange(llList_unsure.shape[0]), size=sampleNum_unsure, replace=False, p=None)

        llList_sure_pos, llList_sure_neg, llList_unsure = llList_sure_pos[sp_idx, ...].copy(), llList_sure_neg[sn_idx, ...].copy(), llList_unsure[us_idx, ...].copy()

        # predict unsure area label with CNN
        gePath_cur = dataPath_curGrid + "/ge_%d_%d_%d.tif" % (ID_grid5d, ID05d, ID01d)
        assert os.path.exists(gePath_cur), "google earth image not found"

        llList_forest_sure, llList_forest_unsure, llList_forest_unsure_offseted = common.getRFSampleList(llList_sure_pos,llList_sure_neg, llList_unsure, model, gePath=gePath_cur, dataLoader=None)

        # append
        llListFS_list.append(llList_forest_sure)
        llListFU_list.append(llList_forest_unsure)
        llListFU_offseted_list.append(llList_forest_unsure_offseted)


    # %%
    predForest_list = []
    predForest_offseted_list = []
    predForest_onlySure_list = []
    for idx, ID05d in enumerate(idList_05d):
        cur_geom05d = grid05d.filterMetadata("ID_05d", "equals", ID05d).first().geometry()

        # # pred with only sure area samples
        # sample_train = common.getRFSampleFC(LC_STN, llListFS_list[idx])
        # classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, variablesPerSplit=9, minLeafPopulation=1, bagFraction=0.5, maxNodes=None, seed=0).train(sample_train, "forest", selVar1)
        # predForest_onlySure = LC_STN.clip(cur_geom05d).select(selVar1).classify(classifier).int8()
        # predForest_onlySure_list.append(predForest_onlySure)

        # # pred with original samples  
        # sample_train = common.getRFSampleFC(LC_STN, llListFS_list[idx] + llListFU_list[idx])
        # classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, variablesPerSplit=9, minLeafPopulation=1, bagFraction=0.5, maxNodes=None, seed=0).train(sample_train, "forest", selVar1)
        # predForest = LC_STN.clip(cur_geom05d).select(selVar1).classify(classifier).int8()
        # predForest_list.append(predForest)

        # pred with offseted samples
        sample_train_offseted = common.getRFSampleFC(LC_STN, llListFS_list[idx] + llListFU_offseted_list[idx])
        classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, variablesPerSplit=9, minLeafPopulation=1, bagFraction=0.5, maxNodes=None, seed=0).train(sample_train_offseted, "forest", selVar1)
        predForest_offseted = LC_STN.clip(cur_geom05d).select(selVar1).classify(classifier).int8()
        predForest_offseted_list.append(predForest_offseted)


    # pred5d = ee.ImageCollection(predForest_list).mosaic().int8()
    # pred5d_onlySure = ee.ImageCollection(predForest_onlySure_list).mosaic().int8()
    pred5d_offseted = ee.ImageCollection(predForest_offseted_list).mosaic().int8()


    # xmin, ymin = grid05d.filterMetadata("ID_05d", "equals", idList_05d[0]).first().getNumber("xmin"), grid05d.filterMetadata("ID_05d", "equals", idList_05d[0]).first().getNumber("ymin")
    # exportRegion = ee.Algorithms.GeometryConstructors.Rectangle(coordinates=[xmin, ymin, xmin.add(0.5), ymin.add(5)])
    # task = ee.batch.Export.image.toAsset(pred5d_offseted, "grid1038", region=exportRegion, scale=30, pyramidingPolicy={"b1":"mode"}, assetId="users/thomasshen99/pred_grid1038")
    # task.start()

    geemap.ee_export_image(pred5d_offseted, "./grid1038.tif", scale=30, region=selected5d.geometry())


    # # %%
    # Map = geemap.Map(center=[56, 26], zoom=12, add_google_map=True)
    # Map.add_basemap('Google Satellite')

    # Map.addLayer(LC_STN.clip(selected5d), vis_params={"max":2000, "min":0, "bands":['B4', 'B3', 'B2']}, name="LC08")

    # Map.addLayer(forest_gong.clip(selected5d), vis_params={"max":1, "min":0, "palette":["FF0000", "00FF00"]}, name="forest_gong")
    # Map.addLayer(pred5d, vis_params={"max":1, "min":0, "palette":["FF0000", "00FF00"]}, name="pred", opacity=1)
    # Map.addLayer(pred5d_offseted, vis_params={"max":1, "min":0, "palette":["FF0000", "00FF00"]}, name="pred_offseted", opacity=1)
    # Map.addLayer(forest_fuse.clip(selected5d), vis_params={"max":5, "min":0, "palette":["FF0000", "FF0000", "FFFF00", "FFFF00", "00FF00", "00FF00"]}, name="forest_fuse", opacity=1)


    # Map.addLayerControl()
    # Map


    # # %%



