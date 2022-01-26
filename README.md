# Fine-Grained VR Sketching: Dataset and Insights

This is the code project for _Fine-Grained VR Sketching: Dataset and Insights_ published on 3DV 2021.

SkethchyVR dataset are avaiable at 
[Dataset webpage](//surrey.ac.uk/Research/vssp_datasets/still/VRChairSketch/html/index.html)
and [Google Drive](https://drive.google.com/file/d/1nRAoj3BISFytRoapYDRKm9gic9j06dkD/view?usp=sharing) (point cloud).

Here are some samples of the sketches in SkethchyVR:
![1](images/4b495.gif)
![1](images/5bdcd.gif)

# VR Sketch interface
Coming soon!

# Sketch filtering
Demonstration on filtering the original sketches: `tools/Filter original sketch.ipynb`

# Point cloud sampling
Once filtering the original VR sketches, point clouds for training are sampled form the filtered sketch. Script for sampling from sketches and meshes: `tools/gen_pointcloud.py`

# Point cloud Rendering
Render image for point cloud files: `tools/vis_pc_mitsuba.py`

Install MITSUBA first and then replace the `PATH_TO_MITSUBA2` with your path.

# Models for 3D shape retrieval

Train 3D sketch to 3D shape retrieval:
`train_triplet_3dv.py`

Train 2D sketch to 3D shape retrieval:
`train_triplet_view_2d.py`