import ee

def generateGrid(xmin, ymin, xmax, ymax, dx, dy):
    """
    按经纬度范围和格网长宽生成渔网
    xmin: down-left longitude
    ymin: down-left latitude
    xmax: up-right longitude
    ymax: up-right latitude
    """
    xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(dx).add(0.001), dx)
    yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(dy).add(0.001), dy)
    cells = xx.map(lambda x: yy.map(lambda y:ee.Feature(ee.Algorithms.GeometryConstructors.Rectangle([x, y, ee.Number(x).add(dx), ee.Number(y).add(dy)]), 
                                                        {'xmin':ee.Number(x), 
                                                         'ymin':ee.Number(y), 
                                                         'ID':ee.Number(x).subtract(xmin).divide(dx).multiply(yy.length()).add(ee.Number(y).subtract(ymin).divide(dy).round()).round()}))).flatten()
    return ee.FeatureCollection(cells)


def pixelStats(image, geometry):
    "按geometry范围统计森林数量， 输入影像需为0/1二值影像"
    pixelNum = image.rename("pixel").selfMask().reduceRegion(reducer=ee.Reducer.sum(), geometry=geometry, scale=30, maxPixels=1e12)
    return ee.Number(pixelNum.get("pixel")).round()


def maskL8sr(image):
    """
    landsat8去云
    """
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    foo = qa.bitwiseAnd(cloudsBitMask).eq(0)
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
    return image.updateMask(mask).updateMask(foo)


def get0501Grid(grid_5d, unsure_image):
    
    xmin5d, ymin5d = grid_5d.getNumber("llLng"), grid_5d.getNumber("llLat")
    grid_05d = generateGrid(xmin5d, ymin5d, xmin5d.add(5), ymin5d.add(5), 0.5, 0.5)\
                .select(["ID", "xmin", "ymin"], ["ID_05d", "xmin", "ymin"], True)\
                .map(lambda x:ee.Feature(x.set({"ID_5d":grid_5d.getNumber("ID")})))

    grid_01d = grid_05d.map(lambda x:generateGrid(x.getNumber("xmin"), x.getNumber("ymin"), x.getNumber("xmin").add(0.5), x.getNumber("ymin").add(0.5), 0.1, 0.1)\
                                        .select(["ID", "xmin", "ymin"], ["ID_01d", "xmin", "ymin"], True)\
                                        .map(lambda grid:grid.set({'forestnum':pixelStats(unsure_image, grid.geometry())}))\
                                        .sort('forestnum', False).first()\
                                        .set({"ID_5d":x.getNumber("ID_5d"), "ID_05d":x.getNumber("ID_05d")}))
    return grid_05d, grid_01d


def CalcMetrics(productName, valPoints):
    m = valPoints.filterMetadata("Type","equals",0).filterMetadata(productName,"equals",1).size()  #TP
    n = valPoints.filterMetadata("Type","equals",0).filterMetadata(productName,"equals",0).size()  #FP
    p = valPoints.filterMetadata("Type","greater_than",0).filterMetadata(productName,"equals",1).size() #FN
    q = valPoints.filterMetadata("Type","greater_than",0).filterMetadata(productName,"equals",0).size() #TN
    
    matrix = ee.Array([[m,n],[p,q]])
    confMat = ee.ConfusionMatrix(matrix)
    OA = confMat.accuracy()
    kappa = confMat.kappa()
    proAcc = m.divide(m.add(n))
    userAcc = m.divide(m.add(p))
    F1 = m.multiply(2).divide(m.multiply(2).add(p).add(n))
    
    metrics = {"productName": productName, 
               "proAcc": proAcc, 
               "userAcc": userAcc, 
               "kappa": kappa, 
               "OA": OA, 
               "F1": F1}
    return metrics


def getCalcMetricsFunc(valPoints, areaDict):
    def CalcMetrics(productName):
        m = valPoints.filterMetadata("Type","equals",0).filterMetadata(productName,"equals",1).size()  #TP
        n = valPoints.filterMetadata("Type","equals",0).filterMetadata(productName,"equals",0).size()  #FP
        p = valPoints.filterMetadata("Type","greater_than",0).filterMetadata(productName,"equals",1).size() #FN
        q = valPoints.filterMetadata("Type","greater_than",0).filterMetadata(productName,"equals",0).size() #TN
        
        matrix = ee.Array([[m,n],[p,q]])
        confMat = ee.ConfusionMatrix(matrix)
        OA = confMat.accuracy()
        kappa = confMat.kappa()
        proAcc = m.divide(m.add(n))
        userAcc = m.divide(m.add(p))
        F1 = m.multiply(2).divide(m.multiply(2).add(p).add(n))
        
        metrics = {"productName": productName, 
                   "proAcc": proAcc, 
                   "userAcc": userAcc, 
                   "kappa": kappa, 
                   "OA": OA, 
                   "F1": F1, 
                   "forestArea": areaDict.getNumber(productName)}
        return metrics

    return CalcMetrics
    


def CalcArea(imgComp, region):
    areaImg = imgComp.multiply(ee.Image.pixelArea())
    return areaImg.reduceRegion(reducer=ee.Reducer.sum(), geometry=region, scale=30, maxPixels=1e13)


def Balancing(featcol1,featcol2, maxRate=4):
    # featcol1 is the minor size feature collection
    featcolMin = ee.FeatureCollection(ee.Algorithms.If(featcol1.size().lt(featcol2.size()),featcol1,featcol2))
    featcolMax = ee.FeatureCollection(ee.Algorithms.If(featcol1.size().lt(featcol2.size()),featcol2,featcol1))
    featcolMin_d_featcolMax = featcolMin.size().divide(featcolMax.size())
    split = featcolMin_d_featcolMax.multiply(maxRate) # 如果原始样本比大于maxRate:1，将被平衡到maxRate:1
    return ee.Algorithms.If(featcolMin_d_featcolMax.lt(1/maxRate), featcolMin.merge(featcolMax.randomColumn('random').filter(ee.Filter.lte('random',split))),featcolMin.merge(featcolMax))


def getBalancedValPoints(valPointsAll, region, maxRate=4):
    valPoints_pos = valPointsAll.filterBounds(region).filterMetadata('Type', 'equals', 0)
    valPoints_neg = valPointsAll.filterBounds(region).filterMetadata('Type', 'greater_than', 0)

    return ee.FeatureCollection(Balancing(valPoints_pos, valPoints_neg, maxRate=maxRate))

