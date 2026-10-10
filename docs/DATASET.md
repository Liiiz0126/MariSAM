# OMF vector annotation release

The planned public resource is a labels-only vector annotation dataset associated with separately licensed PlanetScope scenes. It does not redistribute source imagery, reflectance values, or account credentials. This guide is a proposed release schema, not a statement that the annotations have already been published.

## Files and attributes

Distribute a complete shapefile bundle (`.shp`, `.shx`, `.dbf`, `.prj`, and preferably `.cpg`) rather than only the `.shp` component. Preserve stable object identifiers and polygons in a documented CRS. GeoPackage may additionally be offered for long field names, multiple layers, and Unicode support.

Suggested shapefile fields, each no more than ten characters:

| Field | Meaning |
| --- | --- |
| obj_id | Stable object identifier |
| class_id | 1 = raft, 2 = cage |
| region_id | Study-region identifier |
| scene_id | Exact Planet scene identifier |
| acq_date | Scene acquisition date |
| item_type | Planet catalog item type |
| asset_type | Exact product/asset type used for annotation |
| ann_ver | Annotation version |
| split_id | Optional published benchmark partition |

Where one polygon corresponds to multiple scenes or a mosaic, use a separate object-to-scene CSV rather than truncating or concatenating long identifiers into one field. Document source scenes, mosaic order, acquisition dates, resampling, processing level, band order, image-grid transform, and all alignment changes needed to reproduce the annotated image.

## Annotation coverage and benchmark split

Positive polygons alone do not show which surrounding pixels were reviewed. Release a reviewed-coverage polygon layer so users can distinguish inspected background from unknown/unannotated areas. Identify excluded clouds, nodata, and ambiguous areas where applicable.

Users may rasterize and tile the polygons at different sizes for their own models. For comparison with the article, additionally provide the original train/validation area polygons and the five geographic holdout region boundaries. Publish tile manifests or georeferenced windows if exact patch-level reproduction is required. The flexible labels need not be locked to 256-by-256 inputs, but the original benchmark should remain recoverable.

Users must obtain matching imagery through their own authorized Planet access. Scene identifiers are retrieval metadata, not a redistribution license or a guarantee that every account can access the scene. Check the actual provider agreement before release. For E&R Basic guidance, see https://support.planet.com/hc/en-us/articles/22540409084701-Dataset-Sharing-Guidelines-for-the-Education-and-Research-Basic-Program .
