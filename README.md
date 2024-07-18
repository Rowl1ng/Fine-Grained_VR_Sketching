# Fine-Grained VR Sketching: Dataset and Insights

This is the code project for _Fine-Grained VR Sketching: Dataset and Insights_ published on 3DV 2021.

Paper Link: [[Paper]](https://arxiv.org/abs/2209.10008) [[Supplemental]](https://drive.google.com/file/d/1JXGO1s8pyT7YR26zruDevJwO2XJ4MAE1/view?usp=sharing)

SkethchyVR dataset are avaiable at 
[Dataset webpage](https://cvssp.org/data/VRChairSketch/)
and [Google Drive](https://drive.google.com/file/d/1nRAoj3BISFytRoapYDRKm9gic9j06dkD/view?usp=sharing) (point cloud only).

Here are some samples of the shape sketch pairs in SkethchyVR:

![1](images/4b495_shape.gif)![1](images/4b495.gif)

![2](images/5bdcd_shape.gif)![2](images/5bdcd.gif)

# VR Sketch interface
The dataset used in this project is collected with a VR sketching interface called [SketchyVR](https://github.com/Rowl1ng/SketchyVR) which allow participants to sketch inmmersively using VR headsets and handles like Oculus Rift.

# Sketch filtering
Demonstration on filtering the original sketches: `tools/Filter original sketch.ipynb`

# Point cloud sampling
Once filtering the original VR sketches, point clouds for training are sampled form the filtered sketch. Script for sampling from sketches and meshes: `tools/gen_pointcloud.py`

# Point cloud Rendering
Render image for point cloud files: `tools/vis_pc_mitsuba.py`

Install MITSUBA first and then replace the `PATH_TO_MITSUBA2` with your path.

# Models for 3D shape retrieval

Train 3D sketch based 3D shape retrieval:
`train_triplet_3dv.py`

Train 2D sketch based 3D shape retrieval:
`train_triplet_view_2d.py`

The `val.txt` in the published dataset only includes 101 chair models from **ShapeNetCore**. To make the validation more reliable, I added chair shapes from **ModelNet10**, resulting in a new list file, `val_shape.txt`. Therefore, `val_shape.txt` contains many more chair model names from **ModelNet10** compared to `val.txt`. You can choose which one to use based on your needs. I have uploaded all the list files, including val_shape.txt, [here](https://drive.google.com/file/d/12lz2cfG3bMGuaqGUK0nwEhyPgUz4z_nu/view?usp=sharing). Additionally, the required chair shapes from **ModelNet10** can be downloaded from [this link](https://drive.google.com/file/d/13W3YOOp_qgUYhNflU6L5XKVp6RmpsmuN/view?usp=sharing).

# Cite
Please cite our work if you find it useful:

```
@inproceedings{luo2021fine,
  title={Fine-Grained VR Sketching: Dataset and Insights.},
  author={Luo, Ling and Gryaditskaya, Yulia and Yang, Yongxin and Xiang, Tao and Song, Yi-Zhe},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={1003--1013},
  year={2021},
  organization={IEEE}
}
```