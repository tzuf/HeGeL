
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import box, mapping, LineString, LinearRing
from shapely.geometry.base import BaseGeometry
import shapely.wkt
import numpy as np
import sys


#
# def get_center_from_s2cellids(
#   s2cell_ids):
#   """Returns the center latitude and longitude of s2 cells.
#   Arguments:
#     s2cell_ids: array of valid s2 cell ids. 1D array
#   Returns:
#     a list of shapely points of shape the size of the cellids list.
#   """
#   prediction_coords = []
#   for s2cellid in s2cell_ids:
#     s2_latlng = s2.S2CellId(int(s2cellid)).ToLatLng()
#     lat = s2_latlng.lat().degrees()
#     lng = s2_latlng.lng().degrees()
#     prediction_coords.append([lat, lng])
#   return np.array(prediction_coords)
#
# def predictions_to_points(
#   preds,
#   label_to_cellid):
#   try:
#     default_cell = list(label_to_cellid.values())[0]
#   except:
#     print (f"label_to_cellid: {label_to_cellid}")
#   cellids = []
#   for label in preds:
#     cellids.append(label_to_cellid[label] if label in label_to_cellid else default_cell)
#   coords = get_center_from_s2cellids(cellids)
#   return coords

def list_arrays_from_str_geometry(geometry_str, max_size=300):

    geometry = shapely.wkt.loads(geometry_str)

    assert isinstance(geometry, Polygon) or isinstance(geometry, LineString) or isinstance(geometry, Point), f"geometry: {geometry}"
    if isinstance(geometry, Polygon):
      xx, yy = geometry.exterior.coords.xy
    else:
      xx, yy = geometry.coords.xy

    list_arrays = [np.array((x, y)) for x, y in zip(xx, yy)]

    size_padding = max_size-len(list_arrays)
    assert size_padding>=0

    padded = np.pad(np.vstack(list_arrays), ((0,size_padding),(0,0)), mode='constant')
    assert padded.shape[0] == max_size
    return padded

def point_from_str_coord_xy(coord_str: str):
  '''Converts coordinates in string format (latitude and longtitude) to Point.
  E.g, of string '(-74.037258, 40.715865)' or '[-74.037258, 40.715865]' or 'POINT(-74.037258 40.715865)'.
  Arguments:
    coord: A lng-lat coordinate to be converted to a point.
  Returns:
    A point.
  '''
  list_coords_str = coord_str.replace("POINT", "").replace("(", "").replace(")", "").split(',')
  if len(list_coords_str) == 1:
    list_coords_str = list_coords_str[0].split(' ')

  list_coords_str = [x for x in list_coords_str if x]
  coord = list(map(float, list_coords_str))

  return Point(coord[0], coord[1])

#
# def cellid_from_point(point: Point, level: int):
#   '''Get s2cell covering from shapely point (OpenStreetMaps Nodes).
#   Arguments:
#     point(Point): a Shapely Point to which S2Cells.
#     covering will be performed.
#   Returns:
#     An id of S2Cellsid that cover the provided Shapely Point.
#   '''
#
#   assert isinstance(point, Point), f"Object not a Shapely Point but a type {type(point)}"
#   s2polygon = s2polygon_from_shapely_point(point)
#   cellids = get_s2cover_for_s2polygon(s2polygon, level)
#   if cellids is None:
#     sys.exit("S2cellid covering failed because the point is a None.")
#   cellid = cellids[0]
#   return cellid.id()
#
#
# def s2polygon_from_shapely_point(shapely_point: Point):
#   '''Converts a Shapely Point to an S2Polygon.
#   Arguments:
#     point(Shapely Point): The Shapely Point to be converted to
#     S2Polygon.
#   Returns:
#     The S2Polygon equivelent to the input Shapely Point.
#   '''
#
#   y, x = shapely_point.y, shapely_point.x
#   latlng = s2.S2LatLng.FromDegrees(y, x)
#   return s2.S2Polygon(s2.S2Cell(s2.S2CellId(latlng)))
#
# def cellids_from_polygon(polygon: Polygon, level: int):
#   '''Get s2cell covering from shapely polygon (OpenStreetMaps Ways).
#   Arguments:
#     polygon(Polygon): a Shapely Polygon to which S2Cells.
#     covering will be performed..
#   Returns:
#     A sequence of S2Cells ids that cover the provided Shapely Polygon.
#   '''
#
#   s2polygon = s2polygon_from_shapely_polygon(polygon)
#   s2cells = get_s2cover_for_s2polygon(s2polygon, level)
#   return [cell.id() for cell in s2cells]
#
#
#
# def get_s2cover_for_s2polygon(s2polygon: s2.S2Polygon,
#                               level: int):
#   '''Returns the cellids that cover the shape (point/polygon/polyline).
#   Arguments:
#     s2polygon(S2Polygon): The S2Polygon to which S2Cells covering will be
#     performed.
#   Returns:
#     A sequence of S2Cells that completely cover the provided S2Polygon.
#   '''
#
#   if s2polygon is None:
#     return None
#   coverer = s2.S2RegionCoverer()
#   coverer.set_min_level(level)
#   coverer.set_max_level(level)
#   coverer.set_max_cells(100)
#   covering = coverer.GetCovering(s2polygon)
#   for cell in covering:
#     assert cell.level() == level
#
#   return covering
#
# def s2polygon_from_shapely_polygon(shapely_polygon: Polygon) -> s2.S2Polygon:
#   '''Convert a Shapely Polygon to S2Polygon.
#   Arguments:
#     shapely_polygon(Polygon): The Shapely Polygon to be
#     converted to S2Polygon.
#   Returns:
#     The S2Polygon equivelent to the input Shapely Polygon.
#   '''
#
#   # Filter where shape has no exterior attributes (e.g. lines).
#   if not hasattr(shapely_polygon.buffer(0.00005), 'exterior'):
#     return
#   else:
#     # Add a small buffer for cases where cover doesn't work.
#     list_coords = list(shapely_polygon.buffer(0.00005).exterior.coords)
#
#   # Get list of points.
#   s2point_list = list(map(s2point_from_coord_xy, list_coords))
#   s2point_list = s2point_list[::-1]  # Counterclockwise.
#   return s2.S2Polygon(s2.S2Loop(s2point_list))
#
#
#
# def s2point_from_coord_xy(coord) -> s2.S2Point:
#   '''Converts coordinates (longtitude and latitude) to the S2Point.
#   Arguments:
#     coord: The coordinates given as longtitude and
#     latitude to be converted to S2Point.
#   Returns:
#     The S2Point equivelent to the input coordinates .
#   '''
#
#   # Convert coordinates (lon,lat) to s2LatLng.
#   latlng = s2.S2LatLng.FromDegrees(coord[1], coord[0])
#
#   return latlng.ToPoint()  # S2Point
#
#
