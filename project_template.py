#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
 

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    # chose a voxel alos known as a leaf
    # change from 0.01 to 0.005
    LEAF_SIZE = 0.005

    # call the filter function to obtain the resultant downsample point cloud
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_downsample = vox.filter()


    # Outlier Removal Filter 
    # Creating a filter object: 
    outlier_filter = cloud_downsample.make_statistical_outlier_filter() 
    # Set the number of neighboring points to analyze for any given point 
    # previous  from 5 to 8
    outlier_filter.set_mean_k(8) 
    # Set threshold scale factor 
    # changed from 0.1 to 0.3
    x = 0.3 
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier 
    outlier_filter.set_std_dev_mul_thresh(x) 
    # Ccall the filter function for magic 
    outliers_removed = outlier_filter.filter() 


    # TODO: PassThrough Filter
    passthrough = outliers_removed.make_passthrough_filter()
    # assign axis and range to the passthrough filter
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    passthrough_z = passthrough.filter()
    # Limiting on the Y axis too to avoid having the bins recognized as snacks 
    passthrough = passthrough_z.make_passthrough_filter() 
    # Assign axis and range to the passthrough filter object. 
    # change from y to x
    filter_axis = 'x' 
    passthrough.set_filter_field_name(filter_axis) 
    # change min from -0.45 to 0.34
    # change max from +0.45 to 1.0
    axis_min = 0.34 
    axis_max = 1.0 
    passthrough.set_filter_limits(axis_min, axis_max) 
    
    #use the filter function to obtain the resultant point cloud
    cloud_passthrough = passthrough.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_passthrough.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Set max distance for a point to be consedered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    # Exctract inliers - tabletop
    cloud_table = cloud_passthrough.extract(inliers, negative=False)
    # Extract outliers - objects
    cloud_objects = cloud_passthrough.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    
    # Set tolerance for extraction
    ec.set_ClusterTolerance(0.02)
    # change from 10 to 40
    ec.set_MinClusterSize(40)
    # change 2500 to 4000
    ec.set_MaxClusterSize(4000)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # accept new cloud containing all clusters, each with a unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
 
   # TODO: Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_cloud_passthrough = pcl_to_ros(cloud_passthrough)
    ros_outliners_removed = pcl_to_ros(outliers_removed)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_cloud_pub.publish(ros_cluster_cloud)
    pcl_cloud_passthrough_pub.publish(ros_cloud_passthrough)
    pcl_outliners_removed_pub.publish(ros_outliners_removed)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # convert the cluster from pcl to ROS
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        # extract the histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))

        # get the label
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        # Publish a label into RViz
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels),detected_objects_labels))
    detected_objects_pub.publish(detected_objects)
 
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()

    try:
      pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
      pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    WORLD_ID = 3
    test_scene_number = Int32()
    test_scene_number.data = WORLD_ID
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    my_dict_list = []

    # TODO: Get/Read parameters
    object_list_parm = rospy.get_param('/object_list')
    dropbox_parm_list = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    object_parm_dict = {}
    for idex in range(0, len(object_list_parm)):
        object_parm_dict[object_list_parm[idex]['name']] = object_list_parm[idex]
    
    dropbox_parm_dict = {}
    for idex in range(0, len(dropbox_parm_list)):
        dropbox_parm_dict[dropbox_parm_list[idex]['group']] = dropbox_parm_list[idex]

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for object in object_list:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]

        # Get configuration parameters for this kind of object
        object_parm = object_parm_dict[object.label]
        
        # Dropbox parameters
        dropbox_parm = dropbox_parm_dict[object_parm['group']]

        object_name.data = str(object.label)

        # Create pick pose for this object
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])
        pick_pose.orientation.x = 0.0
        pick_pose.orientation.y = 0.0
        pick_pose.orientation.z = 0.0
        pick_pose.orientation.w = 0.0

        # TODO: Create 'place_pose' for the object
        # use location of drop box plus an incremental offset - do not pile all objects at this same location
        position = dropbox_parm['position'] + np.random.rand(3)/10
        place_pose.position.x = float(position[0])
        place_pose.position.y = float(position[1])
        place_pose.position.z = float(position[2])
        place_pose.orientation.x = 0.0
        place_pose.orientation.y = 0.0
        place_pose.orientation.z = 0.0
        place_pose.orientation.w = 0.0

       
        # TODO: Assign the arm to be used for pick_place
        arm_name.data = str(dropbox_parm['name'])

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_number, arm_name, object_name, pick_pose, place_pose)
        my_dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            #resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            #print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml("output_"+str(WORLD_ID)+".yaml", my_dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_cloud_pub = rospy.Publisher("/pcl_cluster_cloud", PointCloud2, queue_size=1)
    pcl_cloud_passthrough_pub = rospy.Publisher("/pcl_cloud_passthrough", PointCloud2, queue_size=1)
    pcl_outliners_removed_pub = rospy.Publisher("/pcl_outliners_removed", PointCloud2, queue_size=1)

    # Add two new Publishers for object recognition information
    # call them object_markers_pub and detected_objects_pub
    # Have them publish to "/object_marker" and "/detected_objects"
    # Message types "Marker" and DetectedObjectsArray"
    object_markers_pub = rospy.Publisher("object_marker", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
