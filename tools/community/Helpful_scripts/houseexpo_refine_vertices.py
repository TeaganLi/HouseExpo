"""
HouseExpo 2D Floor-Plan Vertex Refiner
--------------------------------------

Purpose
-------
Clean up polygon vertex lists from HouseExpo-style JSON files. The script:
1) Removes later-occurring duplicate vertices (keeps the first occurrence),
2) Snaps nearly-equal X/Y coordinates to a canonical value via mapping,
3) Drops consecutive duplicates introduced by snapping,
4) Removes collinear triples (purely horizontal or vertical runs),
5) Plots Original vs Filtered polygons for quick visual QA.

I/O
---
Input  : A folder containing *.json files with a structure like:
         { "verts": [[x1, y1], [x2, y2], ...] }
Output : No files are modified by default. The script prints and plots results.
         (Code to write cleaned vertices back to JSON is included but commented out.)

Notes
-----
- Uses a small tolerance `TOL` for floating-point comparisons.
- Appends the first vertex to the end of the filtered list.

"""

import os
import json
import matplotlib.pyplot as plt


# Specify your input folder where the original JSON files are located
folder_path = "path/to/original/jsons/folder"  # Change this to your actual folder

# Specify your output folder where the filtered JSON files will be saved
output_path = "path/to/filtered/jsons/disred/lcoation" # Change this to your actual folder
os.makedirs(output_path, exist_ok=True)


TOL = 1e-6

def almost_equal(a, b, tol=TOL):
    """
    Return True if two floats are within `tol` of each other.

    Parameters
    ----------
    a, b : float
        Values to compare.
    tol : float
        Absolute tolerance.

    Returns
    -------
    bool
    """

    return abs(a - b) < tol


def chain_mapping(mapping, key, value):
    """
    Merge coordinate equivalence classes for snapping nearly-equal values.

    This maintains a mapping from raw coordinates to their chosen canonical value.
    If `key` already maps through a chain, we resolve to the final representative
    and make sure the mapping is transitive and consistent.

    Parameters
    ----------
    mapping : dict[float, float]
        Current mapping of original -> canonical coordinate values (either x or y).
    key : float
        The raw coordinate value that should be mapped.
    value : float
        The target canonical coordinate (often a previously kept neighbor).

    Returns
    -------
    float
        The final canonical value chosen for `key`.
    """

    final_value = value
    # Follow existing chains to a final representative
    while final_value in mapping:
        final_value = mapping[final_value]

    if mapping.get(key) == final_value:
        # Already consistent
        return final_value
    elif key in mapping:
        # Key already mapped: choose a single representative (max) and collapse chains
        existing_final = mapping[key]
        chosen = max(existing_final, final_value)
        old_val = existing_final
        mapping[key] = chosen

        # If old_val is not already a key and differs from chosen, redirect it to chosen
        if not any(almost_equal(k, old_val) for k in mapping) and not almost_equal(old_val, chosen):
            mapping[old_val] = chosen
        
        # Re-point any values that referenced old_val/final_value to chosen
        for k in list(mapping.keys()):
            if almost_equal(mapping[k], old_val) or almost_equal(mapping[k], final_value):
                mapping[k] = chosen
        return chosen
    else:
        # First time we see this key: record and return
        mapping[key] = final_value
        return final_value


def resolve_mapping(val, mapping):
    """
    Resolve a coordinate through the mapping chain to its canonical value.

    Parameters
    ----------
    val : float
        Coordinate to resolve.
    mapping : dict[float, float]
        Canonicalization mapping.

    Returns
    -------
    float
        Canonical (possibly unchanged) value.
    """

    while val in mapping:
        val = mapping[val]
    return val

def filter_vertices(verts):
    """
    Core filtering/snapper for polygon vertices.

    Steps
    -----
    1) Keep the first vertex; track it as `prev_kept`.
    2) Iterate the list, attempting to keep axis-aligned progress (dx==0 or dy==0).
       If a non-axis-aligned step appears, mark it as `pending_removed`.
    3) If the next point aligns with `pending_removed` in x or y, we
       'snap' coordinates by building x/y mapping (canonicalization),
       then keep `current` and clear the pending marker.
    4) After one pass, apply the accumulated x/y mappings to all kept points.
    5) Remove consecutive duplicates introduced by snapping.
    6) Remove collinear triples (pure horizontal or pure vertical runs).
    7) Return the final list (open polygon; no repeated last vertex).

    Parameters
    ----------
    verts : list[list[float, float]]
        Input vertices (open polygon).

    Returns
    -------
    list[list[float, float]]
        Filtered vertices (open polygon).
    """

    x_mapping = {}
    y_mapping = {}
    new_verts = []
    new_verts.append(verts[0])
    prev_kept = verts[0]
    pending_removed = None

    # Single pass: keep axis-aligned progress; defer/slurp misaligned candidates
    for i in range(1, len(verts)):
        current = verts[i]

        if pending_removed is not None:
            # If current lines up in x OR y with the pending 'corner', we accept and snap
            if almost_equal(current[0], pending_removed[0]) or almost_equal(current[1], pending_removed[1]):
                if almost_equal(current[0], pending_removed[0]):
                    # Snap X to previous-kept X
                    key, value = current[0], prev_kept[0]
                    if key != value and x_mapping.get(key) != value:
                        x_mapping[key] = chain_mapping(x_mapping, key, value)
                        # Drop identity mappings (k==v)
                        x_mapping = {key: value for key, value in x_mapping.items() if key != value}
                else:
                    # Snap Y to previous-kept Y
                    key, value = current[1], prev_kept[1]
                    if key != value and y_mapping.get(key) != value:
                        y_mapping[key] = chain_mapping(y_mapping, key, value)
                        y_mapping = {key: value for key, value in y_mapping.items() if key != value}
                new_verts.append(current)
                prev_kept = current
                pending_removed = None
            else:
                # Still misaligned: keep it pending; try the next one
                pending_removed = current
                continue
        else:
            # Normal case: keep axis-aligned steps; defer diagonal/off-axis corners
            dx, dy = current[0] - prev_kept[0], current[1] - prev_kept[1]
            if almost_equal(dx, 0) or almost_equal(dy, 0):
                new_verts.append(current)
                prev_kept = current
            else:
                pending_removed = current

    # Apply canonical (snapped) X/Y mappings
    corrected_verts = [[resolve_mapping(pt[0], x_mapping), resolve_mapping(pt[1], y_mapping)] for pt in new_verts]

    # Remove consecutive duplicates introduced by snapping
    deduped_verts = []
    for pt in corrected_verts:
        if not deduped_verts or not almost_equal(deduped_verts[-1][0], pt[0]) or not almost_equal(deduped_verts[-1][1], pt[1]):
            deduped_verts.append(pt)

    # Remove collinear triples: skip middle if (prev, curr, next) share X or share Y
    final_verts = []
    n = len(deduped_verts)
    for i in range(n):
        prev, curr, nxt = deduped_verts[i - 1], deduped_verts[i], deduped_verts[(i + 1) % n]
        if almost_equal(prev[1], curr[1]) and almost_equal(curr[1], nxt[1]):
            continue
        if almost_equal(prev[0], curr[0]) and almost_equal(curr[0], nxt[0]):
            continue
        final_verts.append(curr)
    return final_verts


def plot_polygon(vertices, label, color):
    """
    Plot a polygon with markers and a light fill for visual QA.

    Parameters
    ----------
    vertices : list[list[float, float]]
    label : str
    color : str
    """

    x, y = zip(*vertices)
    plt.plot(x, y, marker='o', linestyle='-', label=label, color=color)
    plt.fill(x, y, color=color, alpha=0.2)


def remove_later_duplicates(verts):
    """
    Keep the first occurrence of each vertex; drop later duplicates.

    Parameters
    ----------
    verts : list[list[float, float]]

    Returns
    -------
    list[list[float, float]]
        Deduplicated vertex list (order preserved).
    """

    seen = set()
    filtered_verts = []
    
    for v in verts:
        v_tuple = tuple(v)  # Convert to tuple for hashability
        if v_tuple not in seen:
            filtered_verts.append(v)
            seen.add(v_tuple)
    
    return filtered_verts

# Process all JSON files in a folder
def process_folder(folder_path):
    """
    Process every `*.json` in `folder_path`:
    - Load `verts`
    - Remove later duplicates
    - Filter/snap & simplify via `filter_vertices`
    - Close the polygon (append first vertex) for plotting
    - Plot Original vs Filtered overlays

    Parameters
    ----------
    folder_path : str
        Path to a folder containing HouseExpo-style JSON files.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        original_verts = data["verts"]
        print(f"Original vertices for {file}: {original_verts}")

        # Step 1: drop later duplicates (keep first occurrences)
        vertices = remove_later_duplicates(original_verts)


        # Step 2: filter/snap/simplify
        filtered_verts = filter_vertices(vertices)

        # Close the polygon by repeating the first vertex
        filtered_verts.append(filtered_verts[0])  # Close the polygon
        print(f"Filtered vertices for {file}: {filtered_verts}")

        out_data = dict(data)              # copy everything
        out_data["verts"] = filtered_verts # use EXACTLY the filtered_verts you built

        out_path = os.path.join(output_path, file)
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(out_data, f_out, indent=4)

        print(f"Wrote filtered JSON → {out_path}")
        
        # Visual comparison
        plt.figure(figsize=(8, 6))
        plot_polygon(original_verts, "Original", "red")
        plot_polygon(filtered_verts, "Filtered", "blue")
        plt.legend()
        plt.title(f"Polygon Comparison: {file}")
        plt.show()

process_folder(folder_path)

