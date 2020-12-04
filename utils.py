import ee

def generateGrid(xmin, ymin, xmax, ymax, dx, dy):
    """
    按经纬度范围和格网长宽生成渔网
    xmin: down-left longitude
    ymin: down-left latitude
    xmax: up-right longitude
    ymax: up-right latitude
    """
    xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(dx), dx)
    yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(dy), dy)
    cells = xx.map(lambda x: yy.map(lambda y:ee.Feature(ee.Algorithms.GeometryConstructors.Rectangle([ee.Number(x), ee.Number(y), 
                                                                                                      ee.Number(x).add(ee.Number(dx)), 
                                                                                                      ee.Number(y).add(ee.Number(dy))])))).flatten()
    return ee.FeatureCollection(cells)


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