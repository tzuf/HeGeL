import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from geopy.distance import great_circle
from shapely.ops import nearest_points


_MAX_LOG_HAVERSINE_DIST = np.log(20039 * 1000)  # in meters.
_EPSILON = 1e-5




def get_error_distances(true_polygon_list, pred_points_list):
  """Compute error distance in meters between true and predicted coordinates.

      Args:
      input_file: TSV file containing example_id and true and
        predicted co-ordinates. One example per line.
      eval_logger: Logger object.

      Returns:
      Array of distance error - one per example.
  """
  error_distances = []
  total_examples = 0

  assert len(true_polygon_list) == len(pred_points_list), f"true_polygon_list: {len(true_polygon_list)}, pred_points_list: {len(pred_points_list)}"
  for idx in range(len(true_polygon_list)):
    poly_arr_true = true_polygon_list[idx]
    poly_arr_true = np.array([x.tolist() for x in poly_arr_true if x[0]>30])

    point_pred = pred_points_list[idx]
    point_pred = Point(point_pred[1], point_pred[0])

    try:
      poly_true = Polygon(poly_arr_true)

      p1, _ = nearest_points(poly_true, point_pred)

      err = great_circle((p1.y, p1.x),
                (point_pred.y, point_pred.x)).m

    except:
      if poly_arr_true.shape[0]==1:
        poly_arr_true = Point(poly_arr_true[0])
        err = great_circle((poly_arr_true.y, poly_arr_true.x),
                (point_pred.y, point_pred.x)).m
        
      else:
        print ("problem ", poly_arr_true.shape)

    error_distances.append(err)
    total_examples += 1
  return error_distances




def compute_metrics(error_distances):
  """Compute distance error metrics given an array of error distances.

      Args:
      error_distances: Array of distance errors.
      eval_logger: Logger object.
  """
  num_examples = len(error_distances)

  accuracy_300m = float(
    len(np.where(np.array(error_distances) <= 300.)[0])) / num_examples

  accuracy_1000m = float(
    len(np.where(np.array(error_distances) <= 1000.)[0])) / num_examples

  mean_distance, median_distance, max_error = np.mean(error_distances), np.median(
    error_distances), np.max(error_distances)
  log_distance = np.sort(
    np.log(error_distances + np.ones_like(error_distances) * _EPSILON))
  # AUC for the distance errors curve. Smaller the better.
  auc = np.trapz(log_distance)

  # Normalized AUC by maximum error possible.
  norm_auc = auc / (_MAX_LOG_HAVERSINE_DIST * (num_examples - 1))

  print(
    f"Metrics: \nExact 300 m accuracy : {accuracy_300m}" +
    f"\n1000 m accuracy : {accuracy_1000m}" + f"\nmean error {mean_distance}, " +
    f"\nmedian error {median_distance}\nmax error {max_error}\n" +
    "AUC of error curve {norm_auc}")
    