from trainers.utils.compute_utils import *


def evaluation_metric_rotation(predict_rotation, gt_rotation):
    geodesic_loss = compute_geodesic_distance_from_two_matrices(predict_rotation.view(-1, 3, 3),
                                                                gt_rotation.view(-1, 3, 3)) / pi * 180
    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))

    geodesic_loss_overlap_none = geodesic_loss[gt_distance.view(-1) > (pi / 2)]
    geodesic_loss_overlap_large = geodesic_loss[gt_distance.view(-1) < (pi / 4)]
    geodesic_loss_overlap_small = geodesic_loss[(gt_distance.view(-1) >= pi / 4) & (gt_distance.view(-1) < pi / 2)]

    res_error = {
        "gt_angle": gt_distance / pi * 180,
        "rotation_geodesic_error_overlap_large": geodesic_loss_overlap_large,
        "rotation_geodesic_error_overlap_small": geodesic_loss_overlap_small,
        "rotation_geodesic_error_overlap_none": geodesic_loss_overlap_none,
        "rotation_geodesic_error": geodesic_loss,
    }
    return res_error


def evaluation_metric_rotation_angle(predict_rotation, gt_rotation, gt_rmat1_array, out_rmat1_array):
    batch = predict_rotation.size(0)
    # _, gt_pitch1 = compute_viewpoint_from_rotation_matrix(gt_rmat1_array, batch)
    # gt_rmat2_array = compute_rotation_matrix_from_two_matrices(gt_rotation, gt_rmat1_array.transpose(1,2))
    # gt_yaw, gt_pitch2 = compute_viewpoint_from_rotation_matrix(gt_rmat2_array, batch)
    # gt_pitch = gt_pitch2 - gt_pitch1

    gt_yaw1, gt_pitch1 = compute_viewpoint_from_rotation_matrix(gt_rmat1_array, batch)
    gt_rmat2_array = compute_rotation_matrix_from_two_matrices(gt_rotation, gt_rmat1_array.transpose(1,2))
    gt_yaw2, gt_pitch2 = compute_viewpoint_from_rotation_matrix(gt_rmat2_array, batch)
    gt_yaw = gt_yaw2 - gt_yaw1
    gt_pitch = gt_pitch2 - gt_pitch1

    if out_rmat1_array is None:
        predict_yaw1, predict_pitch1 = compute_viewpoint_from_rotation_matrix(gt_rmat1_array, batch)
        predict_rmat2_array = compute_rotation_matrix_from_two_matrices(predict_rotation, gt_rmat1_array.transpose(1,2))    
    else: 
        predict_yaw1, predict_pitch1 = compute_viewpoint_from_rotation_matrix(out_rmat1_array, batch)
        predict_rmat2_array = compute_rotation_matrix_from_two_matrices(predict_rotation, out_rmat1_array.transpose(1,2))
    predict_yaw2, predict_pitch2 = compute_viewpoint_from_rotation_matrix(predict_rmat2_array, batch)
    predict_yaw = predict_yaw2 - predict_yaw1
    predict_pitch = predict_pitch2 - predict_pitch1
    
    def angle_range(angle):
        while (angle[angle>=pi].size(0)!=0) or (angle[angle<-pi].size(0)!=0):
            angle[angle>=pi] -= 2*pi
            angle[angle<-pi] += 2*pi
        return angle
    yaw_error = torch.abs(angle_range(gt_yaw - predict_yaw))/ pi * 180
    pitch_error = torch.abs(angle_range(gt_pitch - predict_pitch))/ pi * 180

    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))

    yaw_error_overlap_none = yaw_error[gt_distance.view(-1) > (pi / 2)]
    yaw_error_overlap_large = yaw_error[gt_distance.view(-1) < (pi / 4)]
    yaw_error_overlap_small = yaw_error[(gt_distance.view(-1) >= pi / 4) & (gt_distance.view(-1) < pi / 2)]

    pitch_error_overlap_none = pitch_error[gt_distance.view(-1) > (pi / 2)]
    pitch_error_overlap_large = pitch_error[gt_distance.view(-1) < (pi / 4)]
    pitch_error_overlap_small = pitch_error[(gt_distance.view(-1) >= pi / 4) & (gt_distance.view(-1) < pi / 2)]

    res_error = {
        "rotation_yaw_error_overlap_large": yaw_error_overlap_large,
        "rotation_yaw_error_overlap_small": yaw_error_overlap_small,
        "rotation_yaw_error_overlap_none": yaw_error_overlap_none,
        "rotation_yaw_error": yaw_error,
        "rotation_pitch_error_overlap_large": pitch_error_overlap_large,
        "rotation_pitch_error_overlap_small": pitch_error_overlap_small,
        "rotation_pitch_error_overlap_none": pitch_error_overlap_none,
        "rotation_pitch_error": pitch_error
    }
    return res_error
