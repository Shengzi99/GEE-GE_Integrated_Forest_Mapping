"""
    function: 抓取瓦片地图
    author: Jinbao Zhang
    E-mail: kampau@foxmail.com
    date: 2020.10.21
    version: 1.0
"""
from math import floor, pi, log, tan, atan, exp
import math
import urllib.request as ur
import PIL.Image as pil
import io
from threading import Thread, Lock
import numpy as np
from osgeo import gdal
import osr
import ogr
import os

# 在本地运行该爬虫需要进行科学上网，并设置相应的端口
# 这里的 10809 端口为 v2rayN 默认的端口，使用其他软件进行科学上网需要将这里的 10809 更改成对应的端口
# 在 colab 上面运行时，把这两句注释掉
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'

MAP_URLS = {
    "google": "http://mt2.google.cn/vt/lyrs={style}&src=app&x={x}&y={y}&z={z}",
    "amap": "http://wprd02.is.autonavi.com/appmaptile?style={style}&x={x}&y={y}&z={z}",
    "tencent_s": "http://p3.map.gtimg.com/sateTiles/{z}/{fx}/{fy}/{x}_{y}.jpg",
    "tencent_m": "http://rt0.map.gtimg.com/tile?z={z}&x={x}&y={y}&styleid=4",
    "bing": "https://t.ssl.ak.tiles.virtualearth.net/tiles/a{}.jpeg?g=6821"}

COUNT=0
mutex=Lock()


# ----------------------- TIF影像读取和写入 -------------------------
def get_array_from_dataset(ds, left_col, top_row, cols, rows):
    """ Extract data array from raster dataset given the up-left point and array shape

    # Args:
        ds (gdal.Dataset): Raster dataset.
        left_col (int): the column index of the left boundary.
        top_row (int): the  row index of the top boundary.
        cols (int): the columns of data array to extract.
        rows (int): the rows of the data array to extract.

    # Returns:
        np.ndarray (3-d)
    """
    n_cols = ds.RasterXSize
    n_rows = ds.RasterYSize
    cols = min(n_cols - left_col, cols)
    rows = min(n_rows - top_row, rows)
    data = ds.ReadAsArray(left_col, top_row, cols, rows)
    if data.ndim == 2:
        data = np.expand_dims(data, -1)
    else:
        data = np.transpose(data, (1, 2, 0))
    data = np.where(np.isnan(data), 0, data)
    return data


def save_tif_image(data, tif_path, data_type=gdal.GDT_Byte,
                   geo_transform=(0, 1, 0, 0, 0, -1),
                   projection='WGS84', no_data=None):
    """ save data to tiff file.

    # Args:
        data (np.ndarray): the data array to save.
        tif_path (string): the path to save tiff tile.
        data_type (gdalconst): the data type to save.
        geo_transform: the geo-transform parameters, with 6 parameters indicating the
            (up_x, x_size, 0, up_y, 0, y_size).
        projection (string): the spatial coordinate system.
        no_data (int/float): value that will set to no-data.

    """
    assert isinstance(data, np.ndarray), "the data array must be 'np.ndarray'."
    if data.ndim == 2:
        data = np.expand_dims(data, -1)
    assert data.ndim == 3, "the data array must have 2 or 3 dimensions."

    bands = data.shape[-1]
    rows, cols = data.shape[:2]
    data_type = gdal.GDT_Byte if data_type is None else data_type

    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(tif_path):
        os.remove(tif_path)

    out_raster = driver.Create(tif_path, cols, rows, bands, data_type, ['COMPRESS=LZW', 'TILED=YES'])
    out_raster.SetGeoTransform(geo_transform)
    if projection is not None:
        out_raster_srs = osr.SpatialReference()
        if projection == 'WGS84':
            out_raster_srs.SetWellKnownGeogCS(projection)
        else:
            out_raster_srs.ImportFromWkt(projection)
        out_raster.SetProjection(out_raster_srs.ExportToWkt())

    for band_i in range(bands):
        out_band = out_raster.GetRasterBand(band_i + 1)
        if no_data is not None:
            out_band.SetNoDataValue(no_data)
        out_band.WriteArray(data[:, :, band_i])
        out_band.FlushCache()

# --------------------- 地理坐标转像素坐标 --------------------
def geo2imagexy(trans, x, y):
    '''
    Convert geographic coordinates or projected coordinates to pixel row and col
    :param dataset: GDAL dataset
    :param x: geographic coordinate lon or projected coordinate x
    :param y: projected coordinate lat or projected coordinate y
    :return: (row, col)
    '''
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)


# -----------------GCJ02到WGS84的纠偏与互转---------------------------
def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def delta(lat, lon):
    '''
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0   #  a: 卫星椭球坐标投影到平面地图坐标系的投影因子。
    ee = 0.00669342162296594323   #  ee: 椭球的偏心率。
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}


def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False


def gcj_to_wgs(gcjLon,gcjLat):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"],gcjLat - d["lat"])


def wgs_to_gcj(wgsLon,wgsLat):
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon)
    return wgsLon + d["lon"], wgsLat + d["lat"]

# --------------------------------------------------------------


# ------------------wgs84与web墨卡托互转-------------------------
# WGS-84经纬度转Web墨卡托
def wgs_to_macator(x, y):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2


# Web墨卡托转经纬度
def mecator_to_wgs(x, y):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return x2, y2
# -------------------------------------------------------------

# ---------------------------------------------------------
'''
东经为正，西经为负。北纬为正，南纬为负
j经度 w纬度 z缩放比例[0-22] ,对于卫星图并不能取到最大，测试值是20最大，再大会返回404.
山区卫星图可取的z更小，不同地图来源设置不同。
'''
# 根据WGS-84 的经纬度获取谷歌地图中的瓦片坐标
def wgs84_to_tile(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not(isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    w = 85.0511287798 if w > 85.0511287798 else w
    w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2**z
    x = floor(j * num)
    y = floor(w * num)
    return x, y


def tileframe_to_mecatorframe(zb):
    # 根据瓦片四角坐标，获得该区域四个角的web墨卡托投影坐标
    inx, iny =zb["LT"]   #left top
    inx2,iny2=zb["RB"] #right bottom
    length = 20037508.3427892
    sum = 2**zb["z"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right buttom
    # 返回四个角的投影坐标
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res


def tileframe_to_pixframe(zb):
    # 瓦片坐标转化为最终图片的四个角像素的坐标
    out={}
    width=(zb["RT"][0]-zb["LT"][0]+1)*256
    height=(zb["LB"][1]-zb["LT"][1]+1)*256
    out["LT"]=(0,0)
    out["RT"]=(width,0)
    out["LB"]=(0,-height)
    out["RB"]=(width,-height)
    return out

# -----------------------------------------------------------

class Downloader(Thread):
    # multiple threads downloader
    def __init__(self,index,count,urls,datas,update):
        # index 表示第几个线程，count 表示线程的总数，urls 代表需要下载url列表，datas代表要返回的数据列表。
        # update 表示每下载一个成功就进行的回调函数。
        super().__init__()
        self.urls=urls
        self.datas=datas
        self.index=index
        self.count=count
        self.update=update

    def download(self,url):
        HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        header = ur.Request(url,headers=HEADERS)
        err=0
        while(err<3):
            try:
                data = ur.urlopen(header).read()
            except:
                err+=1
                return None
            else:
                return data
        # raise Exception("Bad network link.")

    def run(self):
        for i,url in enumerate(self.urls):
            if i%self.count != self.index:
                continue
            self.datas[i]=self.download(url)
            if mutex.acquire():
                self.update()
                mutex.release()

def geturl(source, x, y, z, style):
    '''
    Get the picture's url for download.
    style:
        m for map
        s for satellite
    source:
        google or amap or tencent
    x y:
        google-style tile coordinate system
    z:
        zoom
    '''
    if source == 'google':
        furl = MAP_URLS["google"].format(x=x, y=y, z=z, style=style)
    elif source == 'amap':
        # for amap 6 is satellite and 7 is map.
        style = 6 if style == 's' else 8
        furl = MAP_URLS["amap"].format(x=x, y=y, z=z, style=style)
    elif source == 'tencent':
        y = 2**z - 1 - y
        if style == 's':
            furl = MAP_URLS["tencent_s"].format(
                x=x, y=y, z=z, fx=floor(x / 16), fy=floor(y / 16))
        else:
            furl = MAP_URLS["tencent_m"].format(x=x, y=y, z=z)
    else:
        raise Exception("Unknown Map Source ! ")

    return furl

def downpics(urls,multi=20):

    def makeupdate(s):
        def up():
            global COUNT
            COUNT+=1
            print("\b"*45 , end='')
            print("DownLoading ... [{0}/{1}]".format(COUNT,s),end='')
            # print("[{0}/{1}]".format(COUNT,s),end='\n')
        return up

    url_len=len(urls)
    datas=[None] * url_len
    if multi <1 or multi >20 or not isinstance(multi,int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks=[Downloader(i,multi,urls,datas,makeupdate(url_len)) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()

    return datas


def getpic_by_location(locfile, start_line, z, source='google', outdir="output_dir", style='s'):
    '''
    输入坐标文件，根据坐标获取瓦片行列号，输出文件，影像类型（默认为卫星图）
    获取区域内的瓦片并自动拼合图像。返回四个角的瓦片坐标
    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("output directory did not exists, create it.")

    infile = open(locfile, "r", encoding='utf-8')

    locline = infile.readline()

    _count = 0
    while _count < start_line:
        infile.readline()
        _count += 1

    while locline:
        try:
            locline = infile.readline()
            row = locline.split(",")[0]
            col = locline.split(",")[1]

            lng = float(locline.split(",")[2])
            lat = float(locline.split(",")[3])
            posx, posy = wgs84_to_tile(lng, lat, z)
            urls = [geturl(source, i, j, z, style) for j in range(posy - 1, posy + 2) for i in
                    range(posx - 1, posx + 2)]

            try:
                datas = downpics(urls, 9)

                outpic = pil.new('RGB', (3 * 256, 3 * 256))
                for i, data in enumerate(datas):
                    picio = io.BytesIO(data)
                    small_pic = pil.open(picio)

                    y, x = i // 3, i % 3
                    outpic.paste(small_pic, (x * 256, y * 256))
                outpic.save(outdir + "/" + row + "_" + col + ".jpg")
            except:
                continue
        except:
            break


def getpic_by_range(x1, y1, x2, y2, z, source='google', outfile="MAP_OUT.png", style='s'):
    '''
    依次输入左上角的经度、纬度，右下角的经度、纬度，缩放级别，地图源，输出文件，影像类型（默认为卫星图）
    获取区域内的瓦片并自动拼合图像。返回四个角的瓦片坐标
    '''
    pos1x, pos1y = wgs84_to_tile(x1, y1, z)
    pos2x, pos2y = wgs84_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("Total number：{x} X {y}".format(x=lenx, y=leny))

    urls = [geturl(source, i, j, z, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]

    datas = downpics(urls)

    print("\nDownload Finished！ Pics Mergeing......")
    outpic = pil.new('RGBA', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):

        picio = io.BytesIO(data)
        small_pic = pil.open(picio)

        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))

    print('Pics Merged！ Exporting......')
    outpic.save(outfile)
    print('Exported to file！')
    global COUNT
    COUNT = 0
    return {"LT":(pos1x,pos1y),"RT":(pos2x,pos1y),"LB":(pos1x,pos2y),"RB":(pos2x,pos2y),"z":z}


def getpic_tif(x1, y1, x2, y2, z, source='google', out_filename='outfile.tif', style='s'):
    '''
        依次输入左上角的经度、纬度，右下角的经度、纬度，缩放级别，地图源，输出文件，影像类型（默认为卫星图）
        获取区域内的瓦片并自动拼合图像。返回四个角的瓦片坐标
        '''
    pos1x, pos1y = wgs84_to_tile(x1, y1, z)
    pos2x, pos2y = wgs84_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("Total number：{x} X {y}".format(x=lenx, y=leny))

    urls = [geturl(source, i, j, z, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]

    datas = downpics(urls)

    print("\nDownload Finished！ Pics Mergeing......")
    outpic = pil.new('RGB', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        if data == None:
            picio = np.zeros((256, 256, 3))
            small_pic =pil.fromarray(picio.astype('uint8')).convert('RGB')
        else:
            picio = io.BytesIO(data)
            small_pic = pil.open(picio)

        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))

    img = np.array(outpic)

    zb = {"LT": (pos1x, pos1y), "RT": (pos2x, pos1y), "LB": (pos1x, pos2y), "RB": (pos2x, pos2y), "z": z}
    Xframe = tileframe_to_mecatorframe(zb)
    lt_lng, lt_lat = mecator_to_wgs(*Xframe['LT'])
    rb_lng, rb_lat = mecator_to_wgs(*Xframe['RB'])

    geotransform = [0, 0, 0, 0, 0, 0]
    geotransform[0] = lt_lng
    geotransform[3] = lt_lat

    cell_size_x = 360 / 2 ** z / 256
    mean_lat = (lt_lat + rb_lat) / 2.0
    cell_size_y = cell_size_x * np.cos(mean_lat / 180 * np.pi)

    geotransform[1] = cell_size_x
    geotransform[5] = -cell_size_y

    if os.path.exists('tmp.tif'):
        os.remove('tmp.tif')

    save_tif_image(img, 'tmp.tif', geo_transform=tuple(geotransform))

    in_ds = gdal.Open('tmp.tif')
    in_band = in_ds.GetRasterBand(1)
    xsize = in_band.XSize
    ysize = in_band.YSize
    geotrans = list(in_ds.GetGeoTransform())
    geotrans[5] = -cell_size_x

    cell_x_y_times = cell_size_x / cell_size_y
    data = in_ds.ReadAsArray(buf_xsize=xsize, buf_ysize=int(ysize / cell_x_y_times))  # 使用更大的缓冲读取影像，与重采样后影像行列对应
    data = np.transpose(data, (1, 2, 0))

    lt_row, lt_col = geo2imagexy(geotrans, x1, y1)
    rb_row, rb_col = geo2imagexy(geotrans, x2, y2)

    lt_row = int(lt_row + 0.5)
    lt_col = int(lt_col + 0.5)
    rb_row = int(rb_row + 0.5)
    rb_col = int(rb_col + 0.5)

    data = data[lt_col:rb_col, lt_row:rb_row, :]
    geotrans[0] = x1
    geotrans[3] = y1
    save_tif_image(data, out_filename, geo_transform=tuple(geotrans))

    in_ds = None
    os.remove('tmp.tif')
    global COUNT
    COUNT = 0


def getpic_tif_by_shape(shape_filename, z, out_dir):
    '''
    输入网格 shapefile 文件，获取每个网格拼合的谷歌影像，其他地图来源未测试
    每个网格的影像保存为一个tif文件
    :param shape_filename:
    :param out_dir:
    :return:
    '''
    finish_id_list = []
    for item in os.listdir(out_dir):
        if os.path.splitext(item)[1] == '.tif':
            Id = item[10:-4]
            finish_id_list.append(Id)

    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    ogr.RegisterAll()

    ds = ogr.Open(shape_filename, 0)
    oLayer = ds.GetLayerByIndex(0)

    # 对图层进行初始化，如果对图层进行了过滤操作，执行这句后，之前的过滤全部清空
    oLayer.ResetReading()

    feature_count = oLayer.GetFeatureCount(0)
    print('feature count = ', feature_count)

    oFeature = oLayer.GetNextFeature()
    # 下面开始遍历图层中的要素
    while oFeature is not None:
        Id = str(oFeature.GetField('Id'))

        if Id in finish_id_list:
            oFeature = oLayer.GetNextFeature()
            continue

        print("\nProcess Id = ", Id)
        oGeometry = oFeature.GetGeometryRef()
        ring = oGeometry.GetGeometryRef(0)
        point_count = ring.GetPointCount()

        lng_list, lat_list = [], []
        for i in range(point_count):
            point = ring.GetPoint(i)
            lng_list.append(point[0])
            lat_list.append(point[1])

        min_lat = min(lat_list)
        max_lat = max(lat_list)

        min_lng = min(lng_list)
        max_lng = max(lng_list)

        getpic_tif(min_lng, max_lat, max_lng, min_lat, z, source='google', style='s',
                   out_filename='{}/googlemap_{}.tif'.format(out_dir, Id))

        oFeature = oLayer.GetNextFeature()


def screen_out(zb,name):
    if not zb:
        print("N/A")
        return
    print("坐标形式：",name)
    print("左上：({0:.5f},{1:.5f})".format(*zb['LT']))
    print("右上：({0:.5f},{1:.5f})".format(*zb['RT']))
    print("左下：({0:.5f},{1:.5f})".format(*zb['LB']))
    print("右下：({0:.5f},{1:.5f})".format(*zb['RB']))


def file_out(zb, file, target="keep", output="file"):
    '''
    zh_in  : tile coordinate
    file   : a text file for ArcGis
    target : keep = tile to Geographic coordinate
             gcj  = tile to Geographic coordinate,then wgs84 to gcj
             wgs  = tile to Geographic coordinate,then gcj02 to wgs84
    '''
    pixframe = tileframe_to_pixframe(zb)
    Xframe = tileframe_to_mecatorframe(zb)
    for i in ["LT", "LB", "RT", "RB"]:
        Xframe[i] = mecator_to_wgs(*Xframe[i])
    if target == "keep":
        pass;
    elif target == "gcj":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = wgs_to_gcj(*Xframe[i])
    elif target == "wgs":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        raise Exception("Invalid argument: target.")

    if output == "file":
        f = open(file, "w")
        for i in ["LT", "LB", "RT", "RB"]:
            f.write("{0[0]:.5f}, {0[1]:.5f}, {1[0]:.5f}, {1[1]:.5f}\n".format(pixframe[i], Xframe[i]))
        f.close()
        print("Exported link file to ", file)
    else:
        screen_out(Xframe, target)


if __name__ == '__main__':

    # 四种抓取方式

    # 第一种：根据坐标点抓取对应所在的瓦片地图，输出为jpg图像，不带坐标信息
    out_dir = "./data/Global_rs/global_type_210_rs/level20"
    locfile = "./data/Global_rs/global_type_210_loc/global_type_210_loc_split_level20.csv"
    getpic_by_location(locfile, 0, 20, source='google', style='s', outdir=out_dir)

    # 第二种，输入左上角和右下角经纬度，得到拼合的瓦片地图，格式为png，并且输出一个带有坐标信息的txt文件，可用于ArcGIS打开
    x = getpic_by_range(112.986413344, 22.3216975059, 113.486595294, 21.8215155557,
               15, source='google', outfile='map_out.png', style='s')
    file_out(x, 'map_out.txt')

    # 第三种，输入左上角和右下角经纬度，得到拼合的瓦片地图，格式为tif
    getpic_tif(112.986413344, 22.3216975059, 113.486595294, 21.8215155557,
               15, source='google', out_filename='map_out.tif', style='s')

    # 第四种，通过网格 shapefile 文件，获取每个网格拼合的谷歌影像，每个网格的影像保存为一个tif文件
    getpic_tif_by_shape(shape_filename='./data/China_fishnet_0_5.shp', z=15, out_dir='./data/googlemap')
