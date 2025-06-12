from typing import Union, OrderedDict
import SimpleITK as sitk
import numpy as np


def get_multilabel_shape_features(
    input_mask: Union[str, sitk.Image], background: int = 0, allFeatures: bool = False
) -> OrderedDict:
    """
    This function calculates the shape features of the input mask.

    Args:
        input_mask (Union[str, SimpleITK.Image]): The input mask.
        background (int, optional): The background label. Defaults to 0.
        allFeatures (bool, optional): Calculate all features at expense of time. Defaults to False.

    Returns:
        dict: The shape features.
    """

    if isinstance(input_mask, str):
        input_mask = sitk.ReadImage(input_mask)

    # initialize the shape features dictionary
    shape_features = OrderedDict()

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.SetBackgroundValue(background)
    shape_filter.ComputePerimeterOn()
    shape_filter.ComputeFeretDiameterOn()
    if allFeatures:
        shape_filter.ComputeOrientedBoundingBoxOn()
    shape_filter.Execute(input_mask)

    full_resolution = 1
    for s in input_mask.GetSpacing():
        full_resolution *= s

    all_labels = shape_filter.GetLabels()
    ## initialize the shape features dictionary
    temp_feature, temp_feature_direction = (), ()
    for _ in range(input_mask.GetDimension()):
        temp_feature += (0,)
    for _ in range(input_mask.GetDimension()):
        temp_feature_direction += temp_feature

    shape_features["ellipsoid_diameter"] = temp_feature
    shape_features["spherical_perimeter"] = 0
    shape_features["spherical_radius"] = 0
    shape_features["perimeter"] = 0
    shape_features["elongation"] = 0
    shape_features["roundness"] = 0
    shape_features["flatness"] = 0
    # shape_features["size"] = 0
    shape_features["number_of_pixels"] = 0
    shape_features["pixels_on_border"] = 0
    shape_features["perimeter_on_border"] = 0
    shape_features["perimeter_on_border_ratio"] = 0
    shape_features["feretDiameter"] = 0
    ## custom implementations
    shape_features["volume"] = 0
    shape_features["surfaceArea"] = 0
    shape_features["compactness1"] = 0
    shape_features["compactness2"] = 0
    shape_features["sphericity"] = 0
    shape_features["sphericalDisproportion"] = 0
    if allFeatures:
        shape_features["orientedBoundingBox_size"] = temp_feature

    if len(all_labels) == 1:
        label = all_labels[0]
        shape_features[
            "ellipsoid_diameter"
        ] = shape_filter.GetEquivalentEllipsoidDiameter(label)
        shape_features[
            "spherical_perimeter"
        ] = shape_filter.GetEquivalentSphericalPerimeter(label)
        shape_features["spherical_radius"] = shape_filter.GetEquivalentSphericalRadius(
            label
        )
        shape_features["perimeter"] = shape_filter.GetPerimeter(label)
        shape_features["elongation"] = shape_filter.GetElongation(label)
        shape_features["roundness"] = shape_filter.GetRoundness(label)
        shape_features["flatness"] = shape_filter.GetFlatness(label)
        # shape_features["size"] = shape_filter.GetPhysicalSize(label)
        shape_features["number_of_pixels"] = shape_filter.GetNumberOfPixels(label)
        shape_features["pixels_on_border"] = shape_filter.GetNumberOfPixelsOnBorder(
            label
        )
        shape_features["perimeter_on_border"] = shape_filter.GetPerimeterOnBorder(label)
        shape_features[
            "perimeter_on_border_ratio"
        ] = shape_filter.GetPerimeterOnBorderRatio(label)
        shape_features["feretDiameter"] = shape_filter.GetFeretDiameter(label)
        ## custom implementations
        shape_features["volume"] = (
            shape_features.get("number_of_pixels") * full_resolution
        )
        shape_features["surfaceArea"] = (
            shape_features.get("perimeter") * full_resolution
        )
        shape_features["compactness1"] = shape_features.get("volume") / (
            shape_features.get("surfaceArea") ** (3.0 / 2.0) * np.sqrt(np.pi)
        )
        shape_features["compactness2"] = (
            (36 * np.pi)
            * (shape_features.get("volume") ** 2.0)
            / (shape_features.get("surfaceArea") ** 3.0)
        )
        shape_features["sphericity"] = (
            36 * np.pi * shape_features.get("volume") ** 2
        ) ** (1.0 / 3.0) / shape_features.get("surfaceArea")
        shape_features["sphericalDisproportion"] = shape_features.get("surfaceArea") / (
            36 * np.pi * shape_features.get("volume") ** 2
        ) ** (1.0 / 3.0)
        if allFeatures:
            shape_features[
                "orientedBoundingBox_size"
            ] = shape_filter.GetOrientedBoundingBoxSize(label)

    elif len(all_labels) > 1:
        # calculated features for the entire mask
        input_mask_threshold = sitk.BinaryThreshold(
            input_mask, lowerThreshold=all_labels[0], upperThreshold=all_labels[-1]
        )
        shape_features = get_multilabel_shape_features(
            input_mask_threshold, background=background, allFeatures=allFeatures
        )
        for label in all_labels:
            shape_features[
                "ellipsoid_diameter_" + str(label)
            ] = shape_filter.GetEquivalentEllipsoidDiameter(label)
            shape_features[
                "spherical_perimeter_" + str(label)
            ] = shape_filter.GetEquivalentSphericalPerimeter(label)
            shape_features[
                "spherical_radius_" + str(label)
            ] = shape_filter.GetEquivalentSphericalRadius(label)
            shape_features["perimeter_" + str(label)] = shape_filter.GetPerimeter(label)
            shape_features["elongation_" + str(label)] = shape_filter.GetElongation(
                label
            )
            shape_features["roundness_" + str(label)] = shape_filter.GetRoundness(label)
            shape_features["flatness_" + str(label)] = shape_filter.GetFlatness(label)
            # shape_features["size_" + str(label)] = shape_filter.GetPhysicalSize(label)
            shape_features[
                "number_of_pixels_" + str(label)
            ] = shape_filter.GetNumberOfPixels(label)
            shape_features[
                "pixels_on_border_" + str(label)
            ] = shape_filter.GetNumberOfPixelsOnBorder(label)
            shape_features[
                "perimeter_on_border_" + str(label)
            ] = shape_filter.GetPerimeterOnBorder(label)
            shape_features[
                "perimeter_on_border_ratio_" + str(label)
            ] = shape_filter.GetPerimeterOnBorder(label)
            shape_features[
                "feretDiameter_" + str(label)
            ] = shape_filter.GetFeretDiameter(label)
            ## custom implementations
            shape_features["volume_" + str(label)] = (
                shape_features.get("number_of_pixels_" + str(label)) * full_resolution
            )
            shape_features["surfaceArea_" + str(label)] = (
                shape_features.get("perimeter_" + str(label)) * full_resolution
            )
            shape_features["compactness1_" + str(label)] = shape_features.get(
                "volume_" + str(label)
            ) / (
                shape_features.get("surfaceArea_" + str(label)) ** (3.0 / 2.0)
                * np.sqrt(np.pi)
            )
            shape_features["compactness2_" + str(label)] = (
                (36 * np.pi)
                * (shape_features.get("volume_" + str(label)) ** 2.0)
                / (shape_features.get("surfaceArea_" + str(label)) ** 3.0)
            )
            shape_features["sphericity_" + str(label)] = (
                36 * np.pi * shape_features.get("volume_" + str(label)) ** 2
            ) ** (1.0 / 3.0) / shape_features.get("surfaceArea_" + str(label))
            shape_features["sphericalDisproportion_" + str(label)] = shape_features.get(
                "surfaceArea_" + str(label)
            ) / (36 * np.pi * shape_features.get("volume_" + str(label)) ** 2) ** (
                1.0 / 3.0
            )
            if allFeatures:
                shape_features[
                    "orientedBoundingBox_size_" + str(label)
                ] = shape_filter.GetOrientedBoundingBoxSize(label)
    else:
        # print("No labels found in the input mask.")
        pass
    return shape_features


def get_brainmask_features(
    input_mask: Union[str, sitk.Image], allFeatures: bool = True
):
    """
    This function calculates the shape features of the input brain mask.

    Args:
        input_mask (Union[str, SimpleITK.Image]): The input mask.
        background (int, optional): The background label. Defaults to 0.
        allFeatures (bool, optional): Calculate all features at expense of time. Defaults to False.

    Returns:
        dict: The shape features.
    """

    if isinstance(input_mask, str):
        input_mask = sitk.ReadImage(input_mask)

    input_mask_array = sitk.GetArrayFromImage(input_mask)
    input_brain_mask_dict, shape_features = OrderedDict(), OrderedDict()

    mask = np.zeros_like(input_mask_array)
    mask += (input_mask_array == 1).astype(mask.dtype)

    input_brain_mask_dict = sitk.GetImageFromArray(mask)
    input_brain_mask_dict.CopyInformation(input_mask)
    # only analyze largest connected component
    component_image = sitk.ConnectedComponent(input_brain_mask_dict)
    sorted_component_image = sitk.RelabelComponent(
        component_image, sortByObjectSize=True
    )
    largest_component_binary_image = sorted_component_image == 1
    shape_features_temp = get_multilabel_shape_features(
        largest_component_binary_image,
        background=0,
        allFeatures=allFeatures,
    )

    # flatten all features into a list
    shape_features = []
    for feature in shape_features_temp.values():
        if isinstance(feature, tuple) or isinstance(feature, list):
            for x in feature:
                shape_features.append(x)
        else:
            shape_features.append(feature)

    return shape_features
