# Helpful scripts for HouseExpo JSONs

This folder contains small utilities that help **clean** and **prepare** HouseExpo-style floor-plan JSONs.

<img width="1863" height="1265" alt="image" src="https://github.com/user-attachments/assets/a6557cbb-c65c-4df5-bd32-e37f7bcf5ba6" />

---

## 1) `houseexpo_refine_vertices.py`

**What it does**
- Cleans the `verts` list in HouseExpo JSONs by:
  - removing later-occurring duplicate points (keeps the first),
  - snapping nearly-equal X/Y values to a canonical value,
  - removing consecutive duplicates introduced by snapping,
  - removing collinear triples (purely horizontal/vertical runs).
- Plots **Original vs Filtered** polygons for quick visual QA.
- Writes **refined JSON copies** with updated `"verts"` to an output folder.

**Why**

HouseExpo layouts can contain SLAM noise and near-duplicate vertices. This pass produces cleaner polygons for downstream tasks (e.g., AGP, isovists, meshing).

**Input format**
  ```json
  {
    "verts": [[x1, y1], [x2, y2], ...],
    "id": "B_2",
    "room_category": { ... },
    "room_num": 6,
    "bbox": { "min": [.., ..], "max": [.., ..] },
    "include": ""
  }
  ```

**Usage**

* Open the script and set:

  * `input_dir` → folder containing original JSON files

  * `output_dir` → where refined JSONs will be written

* The script will:

  * print before/after vertices,
  * Show a quick overlay plot (close the window to continue),
  * Save a JSON copy per file to `output_dir` with `"verts"` replaced by the refined list.
 
**Notes/limitations**
  * Works well on most JSONs, but not all—edge cases may remain.
  * Plots are for sanity checks; comment out plt.show() for large batches.
  * Uses a small float tolerance (1e-6); adjust if your coordinate scale differs.

**Attribution**

  * Dataset: HouseExpo (MIT License).

  * Script shared to support community preprocessing workflows.




