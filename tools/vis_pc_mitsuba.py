import numpy as np
import sys, os, subprocess
import OpenEXR
import Imath
from PIL import Image
# from plyfile import PlyData, PlyElement

PATH_TO_MITSUBA2 = "/scratch/software/mitsuba2/build/dist/mitsuba"  # mitsuba exectuable
# PATH_TO_MITSUBA2 = "mitsuba"  # mitsuba exectuable

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
xml_head = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="224"/>
            <integer name="height" value="224"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>

"""

# I also use a smaller point size
xml_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
    """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def standardize_bbox(pcl, points_per_object):
    # pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    # np.random.shuffle(pt_indices)
    # pcl = pcl[pt_indices]  # n by 3
    pcl = farthest_point_sample(pcl, points_per_object)
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


# only for debugging reasons
def writeply(vertices, ply_file):
    sv = np.shape(vertices)
    points = []
    for v in range(sv[0]):
        vertex = vertices[v]
        points.append("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
    print(np.shape(points))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(vertices), "".join(points)))
    file.close()


# as done in https://gist.github.com/drakeguan/6303065
def ConvertEXRToJPG(exrfile, jpgfile):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (np.abs(rgb[i]) ** (1.0 / 2.4)) - 0.055) * 255.0)

    rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
    # rgb8 = [Image.fromarray(c.astype(int)) for c in rgb]
    Image.merge("RGB", rgb8).save(jpgfile, "JPEG", quality=95)
    # Image.merge("RGB", rgb8).save(jpgfile, "PNG", quality=95)


def main(argv):
    if (len(argv) < 2):
        print('filename to npy/ply is not passed as argument. terminated.')
        return

    pathToFile = argv[1]
    folder = argv[2]
    if folder == '':
        folder = os.path.dirname(pathToFile)
    if '.npy' in pathToFile:
        filename = os.path.basename(pathToFile).split('.npy')[0]
        file_extension = 'npy'
    elif '.txt' in pathToFile:
        filename = os.path.basename(pathToFile).split('.txt')[0]
        file_extension = 'txt'

    print(filename, file_extension)
    # filename = os.path.basename(pathToFile)

    # for the moment supports npy and ply
    if (file_extension == 'npy'):
        pclTime = np.load(pathToFile)
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == 'npz'):
        pclTime = np.load(pathToFile)
        pclTime = pclTime['pred']
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == 'txt'):
        pclTime = np.loadtxt(pathToFile, delimiter=',').astype(np.float32)
        pclTimeSize = np.shape(pclTime)

    # elif (file_extension == '.ply'):
    #     ply = PlyData.read(pathToFile)
    #     vertex = ply['vertex']
    #     (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
    #     pclTime = np.column_stack((x, y, z))
    else:
        print('unsupported file format.')
        return

    if (len(np.shape(pclTime)) < 3):
        pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
        pclTime.resize(pclTimeSize)

    from math import pi, cos, sin
    def pc_normalize(pc, rotate=0):
        centroid = np.mean(pc, axis=0)
        # pc = pc - centroid
        # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        # pc = pc / m * 0.5
        b = pi * 70 / 180
        ones = np.ones((pc.shape[0], 1))
        z = np.concatenate((pc, ones), axis=1)
        rot_x = np.array([[cos(b), -sin(b), 0, 0],
                          [sin(b), cos(b), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]
                          ])
        b = pi * -45 / 180
        rot_y = np.array([[cos(b), 0, -sin(b), 0],
                          [0, 1, 0, 0],
                          [sin(b), 0, cos(b), 0],
                          [0, 0, 0, 1]
                          ])
        b = pi * -30 / 180
        rot_z = np.array([[1, 0, 0, 0],
                          [0, cos(b), -sin(b), 0],
                          [0, sin(b), cos(b), 0],
                          [0, 0, 0, 1]
                          ])

        return z.dot(rot_x).dot(rot_y).dot(rot_z)[:, :3]

    for pcli in range(0, pclTimeSize[0]):
        pcl = pclTime[pcli, :, :]

        pcl = standardize_bbox(pcl, 2048)
        # new_pcl = np.zeros([pcl.shape[0], 3])
        # new_pcl[:, :2] = pcl[:, :]
        # new_pcl = pc_normalize(new_pcl, rotate=90)
        # print(new_pcl.shape)
        # pcl = new_pcl[:, [2, 0, 1]]
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125

        xml_segments = [xml_head]
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xmlFile = ("%s/%s_%02d.xml" % (folder, filename, pcli))

        with open(xmlFile, 'w') as f:
            f.write(xml_content)
        f.close()

        exrFile = ("%s/%s_%02d.exr" % (folder, filename, pcli))
        if (not os.path.exists(exrFile)):
            print(['Running Mitsuba, writing to: ', xmlFile])
            subprocess.run([PATH_TO_MITSUBA2, xmlFile])
            # os.system(r'C:\Users\ll00931\Documents\project\mitsuba2\mitsuba {}'.format(xmlFile))
        else:
            print('skipping rendering because the EXR file already exists')

        png = ("%s/%s_%02d.jpg" % (folder, filename, pcli))

        print(['Converting EXR to JPG...'])
        ConvertEXRToJPG(exrFile, png)


if __name__ == "__main__":
    # main(sys.argv)
    pathToFile = ''
    from glob import glob

    list_file = '/vol/vssp/datasets/multiview/3VS/datasets/3DV_dataset/list/human_test.txt'
    import random
    name_list = [line.rstrip().split(' ')[0] for line in open(list_file)]
    random.shuffle(name_list)
    pc_paths = [f'/vol/vssp/datasets/multiview/3VS/datasets/3DV_dataset/point/sketch/{item}_sketch_0.5.txt' for item in name_list[:10]]

    pc_paths.extend([f'/vol/vssp/datasets/multiview/3VS/datasets/3DV_dataset/point/human_sketch/{item}.txt' for item in name_list[:10]])
    # existed = glob(r'S:\Research\VR_Sketch\mitsuba_view\shape_view\shapenet\*.jpg')
    # pc_paths = [item for item in pc_paths if os.path.basename(item).split('_')[0] != 'chair']

    save_dir = '/scratch/visualization/3dv_20/hs'#'/scratch/dataset/3D_sketch_2021/mitsuba'

    work_info = [['', pathToFile, save_dir] for pathToFile in pc_paths]
    from multiprocessing import Pool

    with Pool(16) as p:
        p.map(main, work_info)
    # main(argv)