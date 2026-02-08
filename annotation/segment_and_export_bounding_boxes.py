import bpy
import os
import json
import re

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
# Output directory for segmented STL files and JSON annotations
EXPORT_FOLDER = os.path.abspath("./exported_restorations")
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# -------------------------------------------------
# 1. DETECT MAIN STL MESH
# -------------------------------------------------
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

if not mesh_objects:
    raise RuntimeError("No mesh objects found in the Blender scene.")

# Assume the largest mesh corresponds to the intraoral STL scan
main_obj = max(mesh_objects, key=lambda o: len(o.data.vertices))

# Extract numeric scan identifier from object name, if present
match = re.search(r"(\d+)", main_obj.name)
scan_id = match.group(1) if match else "unknown"

# -------------------------------------------------
# 2. DETECT AND STANDARDIZE BOUNDING BOX OBJECTS
# -------------------------------------------------
bbox_objects = [
    obj for obj in bpy.data.objects
    if obj.type == 'MESH' and "Cube" in obj.data.name
]

if not bbox_objects:
    raise RuntimeError("No bounding box cube objects detected.")

# Sort bounding boxes to ensure deterministic ordering
bbox_objects.sort(key=lambda o: (o.location.x, o.location.y, o.location.z))

# Rename bounding boxes consistently
for idx, bbox in enumerate(bbox_objects, start=1):
    bbox.name = f"BBOX_{idx}"

# -------------------------------------------------
# 3. SEGMENT AND EXPORT EACH RESTORATION
# -------------------------------------------------
restoration_index = 1

for bbox in bbox_objects:

    # Duplicate the main STL mesh
    bpy.ops.object.select_all(action='DESELECT')
    main_obj.select_set(True)
    bpy.context.view_layer.objects.active = main_obj
    bpy.ops.object.duplicate()

    dup = bpy.context.selected_objects[0]
    dup.name = f"TEMP_{scan_id}_{restoration_index}"

    # Apply transforms to ensure correct Boolean operation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Apply Boolean intersection using the bounding box
    bool_mod = dup.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = bbox

    bpy.context.view_layer.objects.active = dup
    bpy.ops.object.modifier_apply(modifier="Boolean")

    # Skip empty results
    if len(dup.data.polygons) == 0:
        bpy.data.objects.remove(dup)
        continue

    # -------------------------------------------------
    # EXPORT SEGMENTED STL
    # -------------------------------------------------
    output_name = f"{scan_id}_{restoration_index}"
    stl_path = os.path.join(EXPORT_FOLDER, f"{output_name}.stl")

    bpy.ops.object.select_all(action='DESELECT')
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup

    bpy.ops.wm.stl_export(
        filepath=stl_path,
        export_selected_objects=True
    )

    # -------------------------------------------------
    # EXPORT JSON ANNOTATION
    # -------------------------------------------------
    loc = bbox.location
    dim = bbox.dimensions

    annotation = {
        "scan_id": scan_id,
        "restoration_id": restoration_index,
        "bbox_name": bbox.name,
        "output_stl": f"{output_name}.stl",
        "location_mm": {
            "x": loc.x,
            "y": loc.y,
            "z": loc.z
        },
        "dimensions_mm": {
            "x": dim.x,
            "y": dim.y,
            "z": dim.z
        },
        "min_mm": {
            "x_min": loc.x - dim.x / 2,
            "y_min": loc.y - dim.y / 2,
            "z_min": loc.z - dim.z / 2
        },
        "max_mm": {
            "x_max": loc.x + dim.x / 2,
            "y_max": loc.y + dim.y / 2,
            "z_max": loc.z + dim.z / 2
        }
    }

    json_path = os.path.join(EXPORT_FOLDER, f"{output_name}.json")
    with open(json_path, "w") as f:
        json.dump(annotation, f, indent=4)

    # Remove temporary duplicated object
    bpy.data.objects.remove(dup)

    restoration_index += 1
