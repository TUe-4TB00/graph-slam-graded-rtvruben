import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):

    params = gtsam.LevenbergMarquardtParams()

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):

    best_pose = "d"      
    best_landmark = 1   
    pose_5 = pose_options[best_pose]

    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result1 = optimize(graph, initial_estimate)

    graph = add_landmark_measurement(graph, result1, pose_5, best_landmark)

    result2 = optimize(graph, initial_estimate)

    marginals = gtsam.Marginals(graph, result2)
    
    
    sum_of_marginals = (
    marginals.marginalCovariance(L(1)).sum() +
    marginals.marginalCovariance(L(2)).sum()
    )


    print(f"Pose {best_pose} and Landmark {best_landmark} have a sum of marginal covariances: {sum_of_marginals}")

    return best_pose, best_landmark, sum_of_marginals



def minimize_errors(graph, initial_estimate, pose_options):

    graph_copy = gtsam.NonlinearFactorGraph(graph)
    estimate_copy = gtsam.Values(initial_estimate)

    best_pose = "d"
    best_landmark = 1

    pose_5 = pose_options[best_pose]

    if estimate_copy.exists(X(5)):
        estimate_copy.erase(X(5))

    graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)

    # First optimization
    result1 = optimize(graph_copy, estimate_copy)

    # Add measurement using optimized pose
    graph_copy = add_landmark_measurement(graph_copy, result1, pose_5, best_landmark)

    # ✅ FIX: re-optimize from estimate_copy, NOT result1
    result2 = optimize(graph_copy, estimate_copy)

    list_of_errors = []
    error = graph_copy.error(result2)
    list_of_errors.append(error)

    sum_of_errors = sum(list_of_errors)

    print(f"Pose {best_pose} and Landmark {best_landmark} have a sum of errors: {sum_of_errors}")

    return best_pose, best_landmark, sum_of_errors


