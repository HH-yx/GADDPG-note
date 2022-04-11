#!/usr/bin/env python

# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# for testing    
import argparse
import copy
import datetime
import itertools
import os
import os.path as osp
import pprint
import sys
import threading
import time

import _init_paths
import cv2
import IPython
import matplotlib.pyplot as plt
import message_filters
import numpy as np
import rosnode
import rospy
import std_msgs.msg
# try: # ros
import tf
import tf2_ros
import tf.transformations as tra
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from cv_bridge import CvBridge, CvBridgeError
from experiments.config import *
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from OMG.omg.config import cfg as planner_cfg
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tensorboardX import SummaryWriter
from visualization_msgs.msg import Marker, MarkerArray

from core import networks
from core.bc import BC
from core.ddpg import DDPG
from core.env_planner import EnvPlanner
from core.replay_memory import BaseMemory as ReplayMemory
from core.utils import *

lock = threading.Lock()  # 创建锁   避免多个线程保卫同一块数据的时候，产生错误，所以加锁来防止这种问题

# use posecnn layer for backprojection  使用posecnn层进行反投影
import posecnn_cuda
# graspnet
import tensorflow
from joint_listener import JointListener
# for real robot
from lula_franka.franka import Franka
from moveit import MoveitBridge

sys.path.insert(0, '6dof-graspnet')  # sys.path.insert()临时添加搜索路径（而不是在本路径下添加库），程序退出后失效 -- 用于临时添加本地库

# set policy mode
GA_DDPG_ONLY = True
GRASPNET_ONLY = False
COMBINED = False
RANDOM_TARGET = False
USE_LOOK_AT = False
CONTACT_GRASPNET = False
PUT_BIN = False

# contact graspnet  关联GraspNet
from grasp_estimator import GraspEstimator, get_graspnet_config, joint_config

if CONTACT_GRASPNET:  # 是否要关联GraspNet
    # sys.path.insert()临时添加搜索路径（而不是在本路径下添加库），程序退出后失效 -- 用于临时添加本地库
    sys.path.insert(0, 'contact_graspnet')
    sys.path.insert(0, 'contact_graspnet/contact_graspnet')
    import config_utils
    from contact_grasp_estimator import GraspEstimator as GraspEstimatorContact
    from inference_edit import \
        get_graspnet_config as get_graspnet_config_contact


def compute_look_at_pose(pose_listener, center_object, angle, distance, psi=0):
    """
        compute look at pose according to object pose  根据对象姿势计算注视姿势
        param pose_listener:  定义的监听器
        param center_object: 对象中心
        param angle: 角度
        param distance: 距离
        param psi: ψ y轴旋转角
    """
    # find the hand camera to hand transformation  找到手部摄像机进行手动变换
    try:
        # lookupTransform()第一个参数是目标坐标系, 第二个参数是源坐标系, 第三个参数用于设置查找哪个时刻的坐标变换，第四个参数用于设置查找坐标变换的超时时长
        # lookupTransform()的功能：获得两个坐标系之间转换的关系，包括旋转和平移
        tf_pose = pose_listener.lookupTransform('measured/camera_color_optical_frame', 'measured/right_gripper', rospy.Time(0))
        pose_camera = make_pose(tf_pose)  # 将tf_pose转换为pose 4×4矩阵
    except (tf2_ros.LookupException,  # 查找异常
        tf2_ros.ConnectivityException,  # 连接异常
        tf2_ros.ExtrapolationException):  # 推断异常 
        pose_camera = None

    if pose_camera is not None:  # 找到手部摄像机
        pose_camera[:3, :3] = np.eye(3)  # pose[:3, :3] 部分初始化为对角矩阵
        pose_camera[:3, 3] *= -1
    else:  # 没找到手部摄像机
        print('cannot find camera to hand transformation')

    psi /= 57.3  # ψ y轴旋转角
    theta = angle / 57.3  # θ x轴旋转角(这里将角度转换为弧度)
    r = distance
    # 机械臂位置（绕z轴的基本旋转矩阵）
    position_robot = center_object + np.array([-r * np.cos(theta) * np.cos(psi),
                                               -r * np.cos(theta) * np.sin(psi),
                                                r * np.sin(theta)], dtype=np.float32)
    Z_BG = center_object - position_robot  # 近似于[r * np.cos(psi), r * np.sin(psi), 0]
    Z_BG /= np.linalg.norm(Z_BG)  # 计算第2范数(即r)
    Y_BG = np.array([-np.sin(psi), np.cos(psi), 0], dtype=np.float32)
    X_BG = np.cross(Y_BG, Z_BG)  # 计算两个向量（向量数组）的叉乘。叉乘返回的数组既垂直于a，又垂直于b
    R_BG = np.zeros((3, 3), dtype=np.float32)  # 创建3×3矩阵
    R_BG[:, 0] = X_BG
    R_BG[:, 1] = Y_BG
    R_BG[:, 2] = Z_BG

    pose_robot = np.eye(4, dtype=np.float32)
    pose_robot[:3, 3] = position_robot  # 机械臂角度
    pose_robot[:3, :3] = R_BG[:3, :3]  # 机械臂位置

    # adjust for camera offset  调整相机偏移
    if pose_camera is not None:
        pose_robot = np.dot(pose_camera, pose_robot)
    return pose_robot


class ImageListener:

    def __init__(self, agent, graspnet, graspnet_contact):

        franka = Franka(is_physical_robot=True)  # 定义Franka机器人
        self.moveit = MoveitBridge(franka)  # 定义moveit
        self.moveit.retract()

        # self.moveit.close_gripper()
        self.moveit.open_gripper()  # 张开抓手

        self.joint_listener = JointListener()  # 定义关节监听器
        self.pose_listener = tf.TransformListener()  # 定义姿态监听器
        print('sleep a short time')
        rospy.sleep(2.0)
        print('current robot joints')
        print(self.joint_listener.joint_position)  # 打印关节姿态信息

        # lookupTransform()第一个参数是目标坐标系, 第二个参数是源坐标系, 第三个参数用于设置查找哪个时刻的坐标变换，第四个参数用于设置查找坐标变换的超时时长
        # lookupTransform()的功能：获得两个坐标系之间转换的关系，包括旋转和平移
        tf_pose = self.pose_listener.lookupTransform('measured/panda_hand', 'measured/right_gripper', rospy.Time(0))  # 坐标系转换
        self.grasp_offset = make_pose(tf_pose)  # 将tf_pose转换为pose 4×4矩阵
        print('grasp offset', self.grasp_offset)  # 打印抓手偏移

        self.agent = agent  # 定义agent
        self.graspnet = graspnet  # 定义graspnet
        self.graspnet_contact = graspnet_contact  # 定义graspnet关联
        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.im_ef_pose = None
        self.acc_points = np.zeros([4, 0])  # 维度为0代表这个维度是可拓展的(可变的)
        self.depth_threshold = 1.2
        self.table_height = 0.0
        self.initial_joints = initial_joints  # 初始关节
        self.num_initial_joints = initial_joints.shape[0]  # 初始关节数
        self.index_joints = 0
        self.target_obj_id = 1 # target object ID  目标对象ID

        # publish object points for visualization  发布对象点云以进行可视化
        self.empty_msg = PointCloud2()
        self.object_points2_target_pub = rospy.Publisher('/gaddpg_object_points2_target', PointCloud2, queue_size=10)  # 声明节点使用消息类型PointCloud2发布到'/gaddpg_object_points2_target'话题
        self.object_points2_obstacle_pub = rospy.Publisher('/gaddpg_object_points2_obstacle', PointCloud2, queue_size=10)  # 声明节点使用消息类型PointCloud2发布到'/gaddpg_object_points2_obstacle'话题

        # initialize a node  初始化节点
        self.label_sub = message_filters.Subscriber('seg_label', Image, queue_size=1)  # Subscriber订阅者过滤器，对ROS订阅的封装函数

        self.hand_finger_point = np.array([ [ 0.,  0.,  0.   , -0.   ,  0.   , -0.   ],
                               [ 0.,  0.,  0.053, -0.053,  0.053, -0.053],
                               [ 0.,  0.,  0.075,  0.075,  0.105,  0.105]])

        self.bin_conf_1 = np.array([0.7074745589850109, 0.361727706885124, 0.38521270434333, 
            -1.1754794559646125, -0.4169872830046795, 1.7096866963969337, 1.654512471818922]).astype(np.float32)

        self.bin_conf_2 = np.array([0.5919747534674433, 0.7818432665691674, 0.557417382701195, 
            -1.1647884021323738, -0.39191044586242046, 1.837464805311654, 1.9150514982533562]).astype(np.float32)

        if cfg.ROS_CAMERA == 'D415':  # 使用相机是D415
            # use RealSense D435  使用深度相机D435
            self.base_frame = 'measured/base_link'  # 基础框架
            camera_name = 'cam_2'
            # 订阅ROS话题，订阅者监听订阅的topic，一旦topic进行广播，订阅者就调用回调函数
            rgb_sub = message_filters.Subscriber('/%s/color/image_raw' % camera_name, Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/%s/aligned_depth_to_color/image_raw' % camera_name, Image, queue_size=1)
            # rospy.wait_for_message() 接收的话题如果没有发布消息，它会一直等待，但是接收到一个消息后，等待结束，会继续执行后面的程序
            msg = rospy.wait_for_message('/%s/color/camera_info' % camera_name, CameraInfo)  # 获取相机相关参数(内参)
            self.camera_frame = 'measured/camera_color_optical_frame'  # 相机框架
            self.target_frame = self.base_frame  # 目标采用框架
        elif cfg.ROS_CAMERA == 'Azure':  # 使用相机是Azure
            self.base_frame = 'measured/base_link'
            # 订阅ROS话题，订阅者监听订阅的topic，一旦topic进行广播，订阅者就调用回调函数
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=1)
            # rospy.wait_for_message() 接收的话题如果没有发布消息，它会一直等待，但是接收到一个消息后，等待结束，会继续执行后面的程序
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)  # 获取相机相关参数(内参)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:  # 使用相机是kinect
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (cfg.ROS_CAMERA)
            # 订阅ROS话题，订阅者监听订阅的topic，一旦topic进行广播，订阅者就调用回调函数
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.ROS_CAMERA), Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.ROS_CAMERA), Image, queue_size=1)
            # rospy.wait_for_message() 接收的话题如果没有发布消息，它会一直等待，但是接收到一个消息后，等待结束，会继续执行后面的程序
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.ROS_CAMERA), CameraInfo)  # 获取相机相关参数(内参)
            self.camera_frame = '%s_rgb_optical_frame' % (cfg.ROS_CAMERA)
            self.target_frame = self.base_frame

        # update camera intrinsics  更新相机内参
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1  # 设置消息队列大小
        slop_seconds = 0.4
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, self.label_sub], queue_size, slop_seconds)  # 使用自适应算法来匹配基于其时间戳的消息
        ts.registerCallback(self.callback_rgbdm)  # 注册回调，可以是多个回调，他们将会按照注册的顺序依次进行调用

        # set global intrinsics and extrinsics  设置全局内参和外参
        global INTRINSICS, EXTRINSICS
        INTRINSICS = intrinsics
        EXTRINSICS = np.zeros([4, 4])  # from camera to end effector  从相机到末端执行器
        EXTRINSICS[:3, 3]  = (np.array([0.05253322227958818, -0.05414890498307623, 0.06035263861136299]))   # camera offset  相机偏移
        EXTRINSICS[:3, :3] = quat2mat([0.7182116422267757, 0.016333297635292354, 0.010996322012974747, 0.6955460741463947])  # 四元数转换为3×3矩阵
        self.remaining_step = cfg.RL_MAX_STEP  # 设置剩余时间步长

        # start publishing thread  开始发布线程
        self.start_publishing_tf()  # 开始发布点云线程
        self.planner = EnvPlanner()
        self.expert_plan = []
        self.standoff_idx = -1
        self.has_plan = False
        self.num_trial = 0
        # threshold to close gripper  抓手抓取阈值
        self.grasp_score_threshold = 0.4


    def compute_plan_with_gaddpg(self, state, ef_pose, vis=False):
        """
        generate initial expert plan
        生成初始专家规划
        """
        joints = get_joints(self.joint_listener)  # 获取关节弧度数据
        gaddpg_grasps_from_simulate_view(self.agent, state, cfg.RL_MAX_STEP, ef_pose)  # 模拟DDPG的视图
        print('finish simulate views')
        # can use remaining timesteps to replan. Set vis to visualize collision and traj  可以使用剩余的时间步重新规划。设置vis以可视化碰撞和轨迹
        self.expert_plan, self.standoff_idx = self.planner.expert_plan(cfg.RL_MAX_STEP, joints, ef_pose, state[0][0], vis=vis)  # 专家规划
        print('expert plan', self.expert_plan.shape)
        print('standoff idx', self.standoff_idx)

        
    def start_publishing_tf(self):  # 开始发布点云线程
        self.stop_event = threading.Event()  # 创建一个事件管理标志
        self.tf_thread = threading.Thread(target=self.publish_point_cloud)  # 使用thread类创建线程，target：run()方法调用的回调对象
        self.tf_thread.start()  # 启动线程，即让线程开始执行


    def publish_point_cloud(self):  # 线程调用对象---发布点云信息
        rate = rospy.Rate(30.)  # 初始化Rate对象，设置速率，通过后面的sleep()可以设定循环的频率
        fields = [  # 构造pointXYZ，定义点云特征
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        while not self.stop_event.is_set() and not rospy.is_shutdown():  # 如何停止线程标志为false,且检测程序是否退出标志为flase
            header = std_msgs.msg.Header()  # 创建header类对象，用于自定义信息
            header.stamp = rospy.Time.now()  # 存储ROS中的时间戳信息
            header.frame_id = self.base_frame  # 数据所在的坐标系名称
            out_xyz = self.acc_points[:3, :].T  # T 转置矩阵
            label = self.acc_points[3, :].flatten()  # flatten() 返回一个折叠成一维的数组

            target_xyz = out_xyz[label == 0, :]  # 目标点云xyz为out_xyz中label=0的xyz
            obj_pc2_target = point_cloud2.create_cloud(header, fields, target_xyz)  # 生成PointCloud2消息类型(目标点云)
            self.object_points2_target_pub.publish(obj_pc2_target)  # 发布目标点云话题，发布者负责发布程序感兴趣的topic 或者通知

            obstacle_xyz = out_xyz[label == 1, :]  # 障碍点云xyz为out_xyz中label=1的xyz
            obj_pc2_obstacle = point_cloud2.create_cloud(header, fields, obstacle_xyz)  # 生成PointCloud2消息类型(障碍点云)
            self.object_points2_obstacle_pub.publish(obj_pc2_obstacle)  # 发布障碍点云话题，发布者负责发布程序感兴趣的topic 或者通知

            # if out_xyz.shape[0] > 0:
            #     print('publish points')
            #     print(out_xyz.shape)
            rate.sleep()  # 根据前面定义的频率进行sleep(进行循环更新)


    def callback_rgbdm(self, rgb, depth, mask):  # 回调函数

        ef_pose = get_ef_pose(self.pose_listener)  # 获取pose
        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)  # 将sensor_msgs/Image消息转化为opencv格式的图像消息
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)  # 将sensor_msgs/Image消息转化为opencv格式的图像消息
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(  # 打印日志
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')  # bgr8：带有蓝绿红色顺序的彩色图像，将sensor_msgs/Image消息转化为opencv格式的图像消息
        mask = self.cv_bridge.imgmsg_to_cv2(mask, 'mono8')  # mono8: 灰度图像，将sensor_msgs/Image消息转化为opencv格式的图像消息

        # rescale image if necessary  如有必要，请重新缩放图像
        # Lirui: consider rescaling to 112 x 112 which is used in training (probably not necessary)  考虑重新缩放为112×112，用于训练（可能不是必要的）
        if cfg.SCALES_BASE[0] != 1:  # SCALES_BASE[0]缩放比例
            im_scale = cfg.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)  # 缩放
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)  # 缩放
            mask = pad_im(cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)  # 缩放
        
        with lock:
            self.im = im.copy()
            self.im_ef_pose = ef_pose.copy()
            self.mask = mask.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id  # 数据所在的坐标系名称
            self.rgb_frame_stamp = rgb.header.stamp  # 存储ROS中的时间戳信息


    def show_segmentation_result(self, color, mask, mask_ids):  # 显示分割结果

        image = color.copy()
        for i in range(len(mask_ids)):
            mask_id = mask_ids[i]
            index = np.where(mask == mask_id)
            x = int(np.mean(index[1]))
            y = int(np.mean(index[0]))
            image = cv2.putText(image, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)  # 画图函数，参数：image为图片，str(i+1)为添加的文字，(x, y)为左上角坐标，字体，字体大小，颜色，字体粗细

        cv2.namedWindow("Display 1")  # 设置为窗口大小不变，保持比例，显示灰度
        cv2.imshow("Display 1", image)  # 创建一个窗口显示图片，Display 1为读入的图片名称，image为读入的图片
        cv2.waitKey(0)  # 等待
        cv2.destroyAllWindows()  # 删除建立的全部窗口

        value = input('Please enter which object to pick up: ')  # 接受一个标准输入数据，提示语：Please enter which object to pick up: 请输入要拾取的对象：
        return int(value)  # 返回要拾取的对象


    def find_target_object(self, depth, mask, mask_ids, ef_pose, remaining_step, vis=False):  # 寻找目标对象

        # select target points  选择目标点云
        target_mask = get_target_mask(self.acc_points)
        points = self.acc_points[:3, target_mask]

        # sample points  采样点云
        points = regularize_pc_point_count(points.T, 1024, use_farthest_point=True).T

        # base to hand  
        points = se3_transform_pc(se3_inverse(ef_pose), points)  # 转换坐标系，机械臂本体坐标系到抓手坐标系

        # hand to camera
        offset_pose = se3_inverse(EXTRINSICS)
        xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]  # 坐标系转换到相机坐标系

        # projection to image  映射到图片(点云坐标xyz-->像素坐标uv)
        p_xyz = INTRINSICS.dot(xyz_points)
        index = p_xyz[2] > 0.03
        p_xyz = p_xyz[:, index]
        xyz_points = xyz_points[:, index]
        x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)

        # bounding box  图片边界
        x1 = np.min(x)
        x2 = np.max(x)
        y1 = np.min(y)
        y2 = np.max(y)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 区域0

        # check labels  检查label
        valid_idx_mask = (x > 0) * (x < mask.shape[1] - 1) * (y > 0) * (y < mask.shape[0] - 1)  # 有效的mask
        labels = mask[y[valid_idx_mask], x[valid_idx_mask]]
        labels_nonzero = labels[labels > 0]
        xyz_points = xyz_points[:, valid_idx_mask]

        # find the marjority label
        if float(len(labels_nonzero)) / float((len(labels) + 1)) < 0.5:
            print('overlap to background')  # 与背景重叠
            target_id = -1
        else:
            target_id = np.bincount(labels_nonzero).argmax()  # target_id为labels_nonzero中出现label最多的值的索引

            # check bounding box overlap  选中边界框重叠
            I = np.where(mask == target_id)  # 找到mask中等于target_id的值的索引
            x11 = np.min(I[1])
            x22 = np.max(I[1])
            y11 = np.min(I[0])
            y22 = np.max(I[0])
            area1 = (x22 - x11 + 1) * (y22 - y11 + 1)  # 区域1

            # 取二者中较大的值
            xx1 = np.maximum(x1, x11)
            yy1 = np.maximum(y1, y11)
            # 取二者中较小的值
            xx2 = np.minimum(x2, x22)
            yy2 = np.minimum(y2, y22)

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (area + area1 - inter)  # 重叠区域计算
            print('overlap', ovr)
            if ovr < 0.3:  # 如果图片与目标重叠区域较小
                target_id = -1

            # projected depth  投影深度
            depths = depth[y[valid_idx_mask], x[valid_idx_mask]]
            # computed depth  计算深度
            z = xyz_points[2, :]
            diff = np.mean(np.absolute(depths - z))  # 计算平均深度差异
            print('mean depth diff', diff)  
            if diff > 0.15:
                target_id = -1

        # if remaining_step == cfg.RL_MAX_STEP - 1 and target_id != -1:
        #    self.acc_points = np.zeros([4, 0])

        if vis:  # 是否可视化 
            # show image  显示图片
            fig = plt.figure()  # 新建figure对象
            ax = fig.add_subplot(1, 1, 1)  # 新建子图1
            plt.imshow(mask)  # 接收一张图像，画出该图
            plt.scatter(x[valid_idx_mask], y[valid_idx_mask], s=10)  # 画散点图
            # plt.show()
            plt.show(block=False)  # 显示画的图
            plt.pause(1)  # 设置图片显示的时长，图形会间隔1s后更新
            plt.close()  # 关闭窗口

        return target_id


    def print_joint(self, joint):
        num = len(joint)
        s = ''
        for i in range(num):
            s += '%.6f, ' % rad2deg(joint[i])
        print(s)


    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels

            处理前景标签
                - 将前景标签映射到{0, 1, ..., K-1}

            @param foreground_labels: 一个[H x W] numpy数组

            @return:foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}  找到唯一的(非负的)前景标签，将他们映射到{0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)  # 去除foreground_labels中的重复数字,并进行排序之后输出。
        mapped_labels = foreground_labels.copy()  # 复制foreground_labels原始数组
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels


    def compute_grasp_object_distance(self, RT_grasp):
        T = RT_grasp[:3, 3].reshape((3, 1))

        # target points
        index = self.acc_points[3, :] == 0
        points = self.acc_points[:3, index]
        n = points.shape[1]

        hand = np.repeat(T, n, axis=1)
        distances = np.linalg.norm(hand - points, axis=0)
        return np.min(distances)


    def run_network(self):

        # sample an initial joint  采样初始关节
        if self.remaining_step == cfg.RL_MAX_STEP:  # 剩余步长等于RL_MAX_STEP
            print('use initial joint %d' % (self.index_joints))
            initial_joints = self.initial_joints[self.index_joints, :]  # 初始关节
            self.moveit.go_local(q=initial_joints, wait=True)  # 机械臂动作到初始关节姿态
            rospy.sleep(1.0)

        with lock:
            if listener.im is None:
                print('no image')
                return
            color = self.im.copy()
            depth = self.depth.copy()
            mask = self.mask.copy()
            im_ef_pose = self.im_ef_pose.copy()
            rgb_frame_id = self.rgb_frame_id  # 数据所在的坐标系名称
            rgb_frame_stamp = self.rgb_frame_stamp  # 存储ROS中的时间戳信息

        print('===========================================')

        # process mask  处理mask
        mask = self.process_label(mask)  # 处理前景label
        mask_ids = np.unique(mask)  # 去除mask中的重复数字,并进行排序之后输出
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = mask_ids.shape[0]
        mask_failure = (num == 0)  # 如果num为0(mask_ids没有数据)，则mask_failure为true

        # no mask for the first frame  第一个坐标系没有mask
        if mask_failure and self.remaining_step == cfg.RL_MAX_STEP:  # 如果mask_failure且剩余时间步长为RL_MAX_STEP
            print('no object segmented')  # 没有分割对象
            raw_input('put objects in the scene?')  # raw_input() 用来获取控制台的输入，提示语：put objects in the scene? 在场景中放置对象？
            return

        count = np.zeros((num, ), dtype=np.int32)
        for i in range(num):
            count[i] = len(np.where(mask == mask_ids[i])[0])  # 满足mask中等于mask_ids[i]的个数

        # show the segmentation  显示分割
        start_time = time.time()  # 获取当前时间的时间戳
        if self.remaining_step == cfg.RL_MAX_STEP:  # 如果剩余时间步长为RL_MAX_STEP
            print('%d objects segmented' % num)
            print(mask_ids)


            if not RANDOM_TARGET:  # 不采取随机对象
                label_max = np.argmax(count)  # 获取mask中出现次数最多的label
                target_id = mask_ids[label_max]  # 拾取对象ID为mask出现最后的那个
            else:
                target_id = self.show_segmentation_result(color, mask, mask_ids)  # 获取要拾取对象ID
                '''
                while True:
                    target_id = np.random.choice(mask_ids)
                    # check number of pixels for the target
                    num_pixels = np.sum(mask == target_id)
                    if num_pixels > 500:
                        print('%d target pixels' % num_pixels)
                        break
                '''
        elif num > 0:
            # data association to find the target id for the current frame  数据关联以查找当前坐标系的目标id
            target_id = self.find_target_object(depth, mask, mask_ids, im_ef_pose, self.remaining_step, vis=False)  # 寻找目标对象id
        else:
            target_id = -1
        self.target_obj_id = target_id
        print('target id is %d' % target_id)  # 打印目标对象id
        print("---select target time %s seconds ---" % (time.time() - start_time))  # 打印获取对象所用时间

        if self.remaining_step == cfg.RL_MAX_STEP and not args.fix_initial_state:
            self.index_joints += 1
            if self.index_joints >= self.num_initial_joints:
                self.index_joints = 0

        # process target mask  处理目标mask
        start_time = time.time()
        mask_background = np.zeros_like(mask)
        mask_background[mask == 0] = 1
        if num > 0:
            # update this for 0 background and 1-N for other target  
            mask_target = np.zeros_like(mask)
            mask_target[mask == target_id] = 1
            # erode target mask  对目标mask进行腐蚀操作
            mask_target = cv2.erode(mask_target, np.ones((7, 7), np.uint8), iterations=3)  # 对输入图片进行腐蚀操作，kernel为np.ones((7, 7), np.uint8)
            num_pixels = np.sum(mask_target)  # 像素数
            print('finish mask, %d foreground pixels' % (num_pixels))
            # build the final mask  创建最终mask
            mask[(mask == target_id) & (mask_target == 0)] = 0  # 腐蚀操作对mask_target数据造成一些变化，使用与操作确定最终的mask
            mask_final = mask.copy()
        else:
            mask_final = np.zeros_like(mask)
        print("---process mask time %s seconds ---" % (time.time() - start_time))

        # compute state  计算状态
        start_time = time.time()
        depth = depth[...,None]
        agg = (not mask_failure) and (self.remaining_step >= cfg.RL_MAX_STEP - 1)
        state, point_background = self.camera_image_to_state( color, depth, mask_final, mask_background, im_ef_pose,
                                            cfg.RL_MAX_STEP - self.remaining_step, 
                                            agg=agg, vis=False)  # 摄像机图像绘制图画，在机器人坐标系下目标点云的分割
        print('after camera image to state', state[0].shape)  # 打印(摄像机拍摄后的图像状态[state[0]: point_state, state[1]: image_state])
        print('background point shape', point_background.shape)  # 打印(背景点shape)
        print("---compute state time %s seconds ---" % (time.time() - start_time))  # 计算状态所用时间

        # compute action  计算动作
        state = [state, None, None, None]  

        # look at target
        if self.remaining_step == cfg.RL_MAX_STEP and USE_LOOK_AT:  # 剩余时间步长等于RL_MAX_STEP且使用look at(注视姿态)
            index = self.acc_points[3, :] == 0
            points = self.acc_points[:3, index]
            center = np.mean(points, axis=1)  # 点云中心
            angle = 60
            T_lookat = compute_look_at_pose(self.pose_listener, center, angle=angle, distance=0.45)  # 根据对象姿势计算注视姿势
            self.moveit.go_local(T_lookat, wait=True)  # 机械臂动作到注视姿态位置(T_lookat)
            self.remaining_step = max(self.remaining_step-1, 1)
            rospy.sleep(0.5)
            return

        # GRASPNET + OMG + GA-DDPG

        # run graspnet  运行graspnet
        if (not self.has_plan and COMBINED) or (GRASPNET_ONLY and not GA_DDPG_ONLY):  # (不使用规划，要结合) or (只使用graspnet，不只使用gaddpg)
            point_state = state[0][0].copy() # avoid aggregation  避免聚合
            print('point_state', point_state.shape)
            target_mask = point_state[3, :] == 0  # 获取目标点的mask
            target_pt = point_state[:3, target_mask].T  # 获取目标点，且进行转置
            print('target_pt', target_pt.shape)

            if CONTACT_GRASPNET:  # 是否联系graspnet   # only for target  只针对目标
                #  pc_full: (493949, 3), pc_colors: (493949, 3), pc_segments: dict (idx: (13481, 3)), local_regions True, filter_grasps True, forward_passes 1
                pc_segments = {'0': target_pt}
                point_full = point_state[:3,6:-500].T  # point_state 0-5:抓手点, 6-5002:状态点云, 5002-5502:机器人点
                print('point_full', point_full.shape)
                # all points. You need to add table point here  所有点，你需要在这里添加表格点
                pred_grasps_cam, scores, contact_pts, _ = self.graspnet_contact.predict_scene_grasps(sess_contact, point_full, 
                                                                                             pc_segments=pc_segments, 
                                                                                             local_regions=True,
                                                                                             filter_grasps=True, 
                                                                                             forward_passes=1)  # 预测场景抓取
                # pred_grasps_cam: dict (idx: (N, 4, 4)), scores: dict (idx: (N, 1)), contact_pts: dict (idx: (N, 3))
                # pred_grasps_cam: 预测抓取, scores: 分数, contact_pts: 关联点
                generated_grasps = pred_grasps_cam['0']  # 生成抓取
                generated_scores = scores['0']  # 生成分数
                print('generated contact grasps', generated_grasps.shape)  # 打印(生成关联抓取)
            else:
                latents = self.graspnet.sample_latents()
                generated_grasps, generated_scores, _ = self.graspnet.predict_grasps(
                    sess,
                    target_pt.copy(),
                    latents,
                    num_refine_steps=10,
                )  # 预测抓取

            # select grasps  挑选抓取
            top_num = 100 # grasp num  抓取动作数量上限
            sorted_idx = list(np.argsort(generated_scores))[::-1]  # 找出最大值位置
            select_grasp  = [generated_grasps[idx] for idx in sorted_idx[:top_num]]  # 获取最大值抓取动作
            select_grasp_score = [generated_scores[idx] for idx in sorted_idx[:top_num]]  # 获取最大值抓取动作分数
            print('mean select grasp score: {:.3f}'.format(np.mean(np.round(select_grasp_score, 3))))  # 该函数遵循四舍五入原则(当整数部分以0结束时，round函数一律是向下取整)
            goal_states = np.array([im_ef_pose.dot(g.dot(rotZ(np.pi / 2))) for g in select_grasp]) # might not need rotate  可能不需要轮换
            print(goal_states.shape)
            if goal_states.shape[0] == 0:
                return

            # use OMG in this repo  使用OMG
            planner_cfg.use_external_grasp = True  # 设置使用外部抓取为true
            planner_cfg.external_grasps = goal_states  # this sets the grasps in base coordinate  这将在基础坐标中设置抓取
            joints = get_joints(self.joint_listener)  # 获取关节弧度数据

            # construct scene points  构建场景点
            num = point_state.shape[1] + point_background.shape[1]  # 点云数量
            scene_points = np.ones((4, num), dtype=np.float32)
            scene_points[:, :point_state.shape[1]] = point_state.copy()
            scene_points[:3, point_state.shape[1]:] = point_background.copy()

            step = 30
            plan, standoff_idx = self.planner.expert_plan(step, joints, im_ef_pose, scene_points, vis=False)  # 获取专家规划
            self.has_plan = True
            print('expert plan', plan.shape)

            # execute plan to standoff  执行对抗计划
            if COMBINED:  # 是否结合
                self.moveit.execute(plan[:standoff_idx-5])  # 按照规划的运动路径控制机械臂运动
                self.remaining_step = 10
                print('*****************switch to gaddpg****************')
                rospy.sleep(1.0)
            else:
                self.moveit.execute(plan[:standoff_idx])  # 按照规划的运动路径控制机械臂运动
                self.moveit.execute(plan[standoff_idx:])  # 按照规划的运动路径控制机械臂运动
                rospy.sleep(1.0)
                if PUT_BIN:  # 是否放入目标箱子里
                    self.put_bin()  # 抓住物体，把物体放入一个设置的目标箱子里
                else:
                    self.retract()  # 物体抬起来，放下物体
                self.acc_points = np.zeros([4, 0])
                self.remaining_step = cfg.RL_MAX_STEP
        else:

            if self.termination_heuristics(state) or self.num_trial >= 5:  # 是否使用深度启发法or使用次数大于等于5
                if self.num_trial >= 5:
                    print('********************trial exceed********************')
                if PUT_BIN:  # 是否放入目标箱子里
                    self.put_bin()  # 抓住物体，把物体放入一个设置的目标箱子里
                else:
                    self.retract()  # 物体抬起来，放下物体
                # reset  重置
                self.acc_points = np.zeros([4, 0])
                self.remaining_step = cfg.RL_MAX_STEP
                self.has_plan = False
                self.num_trial = 0
                return

            # run ga-ddpg  运行ga-ddpg
            print('use ga-ddpg')
            target_state = select_target_point(state)  # 获取gaddpg输入的目标点云--only target points  只有目标点
            action, _, _, aux_pred = self.agent.select_action(target_state, remain_timestep=self.remaining_step)  # 挑选动作
            print('finish network') 
            pose_delta = unpack_action(action)
            ef_pose = get_ef_pose(self.pose_listener)  # 获取末端执行器姿态
            ef_pose = ef_pose.dot(pose_delta)  # 动作姿态
            RT_grasp = ef_pose.dot(self.grasp_offset)  # 右抓手动作姿态
            vis_pose = ef_pose.copy()
            # send_transform(RT_grasp, vis_pose, 'GADDPG_action')
            self.moveit.go_local(RT_grasp, wait=True)  # 机械臂动作到RT_grasp
            print('remaining step: {} aggr. point: {}'.format(self.remaining_step, self.acc_points.shape[1]))
            # raw_input('next step?')
        
            self.remaining_step = max(self.remaining_step-1, 1)
            if self.remaining_step == 1:
                self.remaining_step += 5
                self.num_trial += 1


    def retract(self):
        """
        close finger and lift  合上抓手并将物体抬起来
        """    
        # close the gripper  合上抓手--抓取物体
        self.moveit.close_gripper(force=60)
        rospy.sleep(1.0)

        # lift object  抬起物体
        delta = 0.20
        joints = get_joints(self.joint_listener)  # 获取关节弧度数据
        T = self.moveit.forward_kinematics(joints[:-2])  # 正向运动学方法(maybe 获取最后两关节的位置)
        print('T in retract', T)
        T_lift = T.copy()
        T_lift[2, 3] += delta
        self.moveit.go_local(T_lift, wait=True)  # 机械臂动作到T_lift(机械臂抬起物体位置)--将物体抬到T_lift
        # wait a few seconds  等待2s
        rospy.sleep(2.0)

        # put object down  放下物体
        T_put = T.copy()
        T_put[2, 3] += 0.01
        self.moveit.go_local(T_put, wait=True)  # 机械臂动作到T_put(机械臂放置物体位置)--将物体抬到T_put
        self.moveit.open_gripper()  # 张开抓手--放下物体
        self.moveit.go_local(T_lift, wait=True)  # 机械臂动作到T_lift(机械臂抬起物体位置)--机械臂回到抬起姿态

        if GA_DDPG_ONLY:
            self.moveit.retract()
        else:
            step = 20
            joint_position = get_joints(self.joint_listener)  # 获取关节弧度数据
            end_conf = np.append(self.moveit.home_q, joint_position[7:])  # 最后位置配置参数(初始位置+关节位置)
            traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]  # 根据关节位置和最后放置位置来规划路径
            self.moveit.execute(traj)  # 按照规划的运动路径控制机械臂运动

        raw_input('finished. Try again?')


    # grasp object and put object into a bin with goal conf  抓住物体，把物体放入一个设置的目标箱子里
    def put_bin(self):

        force_before = self.joint_listener.robot_force  # 获取抓取前抓力
        print('force before grasping', force_before)

        # close the gripper  合上抓手
        self.moveit.close_gripper(force=60)  # 合上抓手，指定抓力--抓取物体
        rospy.sleep(0.5)

        # lift object a bit  把物体抬起一点
        delta = 0.05
        joints = get_joints(self.joint_listener)  # 获取关节弧度数据
        T = self.moveit.forward_kinematics(joints[:-2])  # 正向运动学方法(maybe 获取最后两关节的位置)
        print('T in retract', T)
        T_lift = T.copy()
        T_lift[2, 3] += delta
        self.moveit.go_local(T_lift, wait=True)  # 机械臂动作到T_lift(机械臂抬起物体位置)--将物体抬到T_lift

        force_after = self.joint_listener.robot_force  # 获取抓取后抓力
        print('force after grasping', force_after)
        force_diff = np.linalg.norm(force_before - force_after)  # 计算抓力前后差值的第二范数
        print('force diff norm', force_diff)  # 抓力差范数值

        # lift object more  抓取更多物体
        delta = 0.30
        joints = get_joints(self.joint_listener)  # 获取关节弧度数据
        T = self.moveit.forward_kinematics(joints[:-2])  # 正向运动学方法(maybe 获取最后两关节的位置)
        print('T in retract', T)
        T_lift = T.copy()
        T_lift[2, 3] += delta
        self.moveit.go_local(T_lift, wait=True)  # 机械臂动作到T_lift(机械臂抬起物体位置)--将物体抬到T_lift

        # check grasp success  检查是否抓取成功
        joint_position = self.joint_listener.joint_position  # 获取关节位置信息
        print('check success', joint_position)
        if joint_position[-1] > 0.002 or force_diff > 0.5 or force_diff == 0:
            success = True
            print('grasp success')
        else:
            success = False
            print('grasp fail')

        # plan to goal conf  规划目标config
        step = 20
        # success
        if success:
            joint_position = get_joints(self.joint_listener)  # 获取关节弧度数据
            end_conf = np.append(self.bin_conf_1, joint_position[7:])  # 最后位置配置参数(箱子位置1+关节位置)
            traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]  # 根据关节位置和最后放置位置来规划路径
            self.moveit.execute(traj)  # 按照规划的运动路径控制机械臂运动

            joint_position = get_joints(self.joint_listener)  # 获取关节弧度数据
            end_conf = np.append(self.bin_conf_2, joint_position[7:])   # 最后位置配置参数(箱子位置2+关节位置)
            traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]  # 根据关节位置和最后放置位置来规划路径
            self.moveit.execute(traj)  # 按照规划的运动路径控制机械臂运动
            self.moveit.open_gripper()  # 张开抓手

        joint_position = get_joints(self.joint_listener)  # 获取关节弧度数据
        end_conf = np.append(self.moveit.home_q, joint_position[7:])   # 最后位置配置参数(初始位置+关节位置)
        traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]  # 根据关节位置和最后放置位置来规划路径
        self.moveit.execute(traj)  # 按照规划的运动路径控制机械臂运动
        self.moveit.open_gripper()  # 张开抓手


    def bias_target_pc_regularize(self, point_state, total_point_num, target_pt_num=1024, use_farthest_point=True):
        """ 
        目标点云规范化
        """
        target_mask = point_state[3, :] == 0  # 获取目标点index
        target_pt = point_state[:, target_mask]  # 获取目标点
        nontarget_pt = point_state[:, ~target_mask]  # 获取非目标点
        print(target_pt.shape, nontarget_pt.shape)
        if target_pt.shape[1] > 0:
            target_pt = regularize_pc_point_count(target_pt.T, target_pt_num, use_farthest_point).T  # 对target_pt的转置进行采样target_pt_num个点
        if nontarget_pt.shape[1] > 0:
            effective_target_pt_num = min(target_pt_num, target_pt.shape[1])  # 如果出现目标点数量小于目标点采样数target_pt_num时
            nontarget_pt = regularize_pc_point_count(nontarget_pt.T, total_point_num - effective_target_pt_num, use_farthest_point).T  # 对nontarget_pt的转置进行采样total_point_num - effective_target_pt_num个点
        return np.concatenate((target_pt, nontarget_pt), axis=1)  # 将target_pt和nontarget_pt两个矩阵拼接在一起


    # new_points is in hand coordinate  new_points位于抓手坐标系
    # ACC_POINTS is in base  ACC_POINTS处于基坐标系
    def update_curr_acc_points(self, new_points, ef_pose, step):
        """
        Update accumulated points in world coordinate
        更新世界坐标系中的累加点
        """
        new_points = se3_transform_pc(ef_pose, new_points)  # 转换坐标系
        # the number below can be adjusted for efficiency and robustness  以下数字可根据效率和稳健性进行调整
        aggr_sample_point_num = min(int(CONFIG.pt_accumulate_ratio**step * CONFIG.uniform_num_pts), new_points.shape[1])  # sample点的数目
        index = np.random.choice(range(new_points.shape[1]), size=aggr_sample_point_num, replace=False).astype(np.int)  # 岁ij生成sample点下标

        new_points = new_points[:,index]  # 生成采样过后的新的new_points
        print('new points before filtering with table height', new_points.shape)  # 打印(经过表格高度过滤之前的new_points)
        index = new_points[2, :] > self.table_height
        new_points = new_points[:, index]  # 生成经过表格过滤后的new_points
        print('new points {} total point {}'.format(new_points.shape, self.acc_points.shape))  # 打印(经过表格过滤后的new_points，以及初始累加点)

        self.acc_points = np.concatenate((new_points, self.acc_points), axis=1)  # np.concatenate()将两个数组拼接在一起
        self.acc_points = regularize_pc_point_count(self.acc_points.T, 4096, use_farthest_point=True).T  # 对self.acc_points.T进行采样，采样4096个点
        # if it still grows too much, can limit points by call regularize pc point count  如果仍然增长过多,可以通过调用正则化pc点数来限制点数
        # self.planner.expert_plan can also be called with these dense points directly  self.planner.expert_plan也可以用这些密集点直接调用


    def goal_closure(self, action, goal):
        action_2 = np.zeros(7)
        action_2[-3:] = action[:3]
        action_2[:-3] = mat2quat(euler2mat(action[3], action[4], action[5])) # euler to quat

        point_dist = float(agent.goal_pred_loss(torch.from_numpy(goal)[None].float().cuda(), 
                            torch.from_numpy(action_2)[None].float().cuda()))
        print('point dist: {:.3f}'.format(point_dist))
        return point_dist < 0.008        


    def graspnet_closure(self, point_state):
        """
        Compute grasp quality from tf grasp net.   从tf抓取网络计算抓取质量
        """
        score  = self.graspnet.compute_grasps_score(sess,  point_state)  # 计算抓取分数
        print('grasp closure score:', score)  # 抓取闭环分数
        return score > self.grasp_score_threshold # tuned threshold  调节阈值(self.grasp_score_threshold=0.4)


    # point_state is in hand coordinate  point_state处于抓手坐标系
    def process_pointcloud(self, point_state, im_ef_pose, step, agg=True, use_farthest_point=False):
        """
        Process the cluttered scene point_state
        [0 - 6]: random or gripper points with mask -1  
        [6 - 1030]: target point with mask 0   
        [1030 - 5002]: obstacle point with mask 1  
        [5002 - 5502]: robot points with mask 2 can be random or generated with get_collision_points and transform with joint
        处理杂乱场景下的point_state
        [0 - 6]: 随机或者带有mask=-1的抓手点
        [6 - 1030]: 带有mask=0的目标点
        [1030 - 5002]: 带有mask=1的障碍点
        [5002 - 5502]: 带有mask=2的机器人点,可以是随机的,也可以使用get_collision_points生成并经过转换到joint
        """

        # accumulate all point state in base  将所有点状态累加到基坐标系中
        # set the mask 0 as target, 1 as other objects  将mask=0设置为目标,mask=1设置为其他对象
        index_target = point_state[3, :] == self.target_obj_id  # 获取目标对象下标
        index_other = point_state[3, :] != self.target_obj_id  # 获取其他对象下标
        point_state[3, index_target] = 0.  # 将目标对象设置为mask=0
        point_state[3, index_other] = 1.  # 将其他对象设置为mask=1
 
        if agg: 
            self.update_curr_acc_points(point_state, im_ef_pose, step)  # 更新世界坐标系中的累加点

        # base to hand
        inv_ef_pose = se3_inverse(im_ef_pose)  # 矩阵的逆
        point_state = se3_transform_pc(inv_ef_pose, self.acc_points)  # 转换坐标系
        point_state = self.bias_target_pc_regularize(point_state, CONFIG.uniform_num_pts)  # 目标点云规范化

        hand_finger_point = np.concatenate([self.hand_finger_point, np.ones((1, self.hand_finger_point.shape[1]), dtype=np.float32)], axis=0)  # 矩阵拼接(抓手点)
        point_state = np.concatenate([hand_finger_point, point_state], axis=1)  # 矩阵拼接(抓手点 + 状态点[目标点和障碍点])
        point_state_ = point_state.copy()
        point_state_[3, :hand_finger_point.shape[1]] = -1  # 抓手点设为mask=-1
        # ignore robot points make sure it's 6 + 4096 + 500  忽略机器人点确保它是6 + 4096 + 500
        point_state_ = np.concatenate((point_state_, np.zeros((4, 500))), axis=1)  # 矩阵拼接(抓手点 + 状态点[目标点和障碍点] + 机器人点)
        point_state_[3, -500:] = 2  # 机器人点设为mask=2
        return point_state_


    def camera_image_to_state(self, rgb, depth, mask, mask_background, im_ef_pose, step, agg=True, vis=False):
        """
        map from camera images and segmentations to object point cloud in robot coordinate
        摄像机图像绘制图画，在机器人坐标系下目标点云的分割
        mask: 0 represents target, 1 everything else
        mask: w x h x 1
        rgb:  w x h x 3
        depth:w x h x 1
        """
        if vis:  # 是否可视化
            fig = plt.figure(figsize=(14.4, 4.8))  # 新建figure对象
            ax = fig.add_subplot(1, 3, 1)  # 新建子图1
            plt.imshow(rgb[:, :, (2, 1, 0)])  # 接收一张图像，画出该图
            ax = fig.add_subplot(1, 3, 2)  # 新建子图2
            plt.imshow(depth[...,0])  # 接收一张图像，画出该图
            ax = fig.add_subplot(1, 3, 3)  # 新建子图3
            plt.imshow(mask)  # 接收一张图像，画出该图
            plt.show()  # 显示画的图

        mask_target = np.zeros_like(mask)
        mask_target[mask == self.target_obj_id] = 1
        mask_state = 1 - mask_target[...,None]  # target:1，其他为0
        image_state = np.concatenate([rgb, depth, mask_state], axis=-1)  # 矩阵合并
        image_state = image_state.T  # 转置
        
        # depth to camera, all the points on foreground objects
        # backproject depth  反投影深度
        depth_cuda = torch.from_numpy(depth).cuda()  # 从numpy.ndarray创建一个张量depth_cuda
        # 获取相机内参
        fx = INTRINSICS[0, 0]
        fy = INTRINSICS[1, 1]
        px = INTRINSICS[0, 2]
        py = INTRINSICS[1, 2]
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth_cuda)[0].cpu().numpy()  # 向前反投影

        # select points  选择点
        valid = (depth[...,0] != 0) * (mask > 0)
        point_xyz = im_pcloud[valid, :].reshape(-1, 3)
        label = mask[valid][...,None]
        point_state = np.concatenate((point_xyz, label), axis=1).T  # 矩阵合并
        # point_state = backproject_camera_target_realworld_clutter(depth, INTRINSICS, mask)
        print('%d foreground points' % point_state.shape[1])
  
        # filter depth  过滤深度
        index = point_state[2, :] < self.depth_threshold
        point_state = point_state[:, index]

        # camera to hand  相机坐标系-->抓手坐标系
        point_state = se3_transform_pc(EXTRINSICS, point_state)  # 转换坐标系

        # background points  背景点
        valid = (depth[...,0] != 0) * (mask_background > 0)
        point_background = im_pcloud[valid, :].reshape(-1, 3)
        index = point_background[:, 2] < self.depth_threshold
        point_background = point_background[index, :]  # 选择背景点
        if point_background.shape[0] > 0:
            point_background = regularize_pc_point_count(point_background, 1024, use_farthest_point=False)  # 在point_background中采样1024个点
            point_background = se3_transform_pc(EXTRINSICS, point_background.T)  # 转换坐标系

        # accumate points in base, and transform to hand again  在底部精确定位点，并再次转换为手
        point_state = self.process_pointcloud(point_state, im_ef_pose, step, agg)  # 处理杂乱场景下的point_state
        obs = (point_state, image_state)  
        return obs, point_background  # 返回观察值，背景点


    # state points and grasp are in hand coordinate
    def vis_realworld(self, state, rgb, grasp, local_view=True, curr_joint=None):
        """
        visualize grasp and current observation 
        local view (hand camera view with projected points)
        global view (with robot and accumulated points) 
        this can be converted to ros
        """

        ef_pose = get_ef_pose(self.pose_listener)
        if local_view:
            print('in vis realworld local view')
            # base to hand
            points = se3_transform_pc(se3_inverse(ef_pose), self.acc_points)
            rgb = rgb[:,:,::-1]
            rgb = proj_point_img(rgb, INTRINSICS, se3_inverse(EXTRINSICS), points[:3], real_world=True)
            grasp = unpack_pose_rot_first(grasp) # .dot(rotZ(np.pi/2))
            rgb = draw_grasp_img(rgb, grasp, INTRINSICS, se3_inverse(EXTRINSICS), vis=True, real_world=True) 
            # show image
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(rgb)
            plt.show()
        else:
            print('in vis realworld global view')
            # global view
            point_color = [255, 255, 0]
            if curr_joint is None:
                curr_joint = get_joints(self.joint_listener)  # 获取关节弧度数据
                point_color = [0, 255, 0]
            poses_ = robot.forward_kinematics_parallel(
                                wrap_value(curr_joint)[None], offset=True)[0]
            grasp = poses_[7].dot(unpack_pose_rot_first(grasp)) 
            poses = [pack_pose(pose) for pose in poses_]
            line_starts, line_ends = grasp_gripper_lines(grasp[None])

            # green: observation, yellow: simulation, red: cage point
            cage_points_mask, depth_heuristics = self.compute_cage_point_mask( )
            noncage_points = self.acc_points[:3, ~cage_points_mask]
            cage_points = self.acc_points[:3, cage_points_mask]
            rgb = self.planner.planner_scene.renderer.vis(poses, list(range(10)), 
                shifted_pose=np.eye(4),
                interact=2,
                V=np.array(V),
                visualize_context={
                    "white_bg": True,
                    "project_point": [noncage_points, cage_points],
                    "project_color": [[0, 255, 0], [255, 0, 0]],
                    "static_buffer": True,
                    "reset_line_point": True,
                    "thickness": [2],
                    "line": [(line_starts[0], line_ends[0])],
                    "line_color": [[255, 0, 0]],            
                }
            )
        return rgb


    def compute_cage_point_mask(self):
        # points in global cooridnate
        index = self.acc_points[3, :] == 0
        points = self.acc_points[:3, index]

        # base to hand
        ef_pose = get_ef_pose(self.pose_listener)
        inv_ef_pose = se3_inverse(ef_pose)
        point_state = se3_transform_pc(inv_ef_pose, points)  
        
        # 0.11
        cage_points_mask =  (point_state[2] > 0.06)  * (point_state[2] < 0.09) * \
                            (point_state[1] > -0.05) * (point_state[1] < 0.05) * \
                            (point_state[0] > -0.02) * (point_state[0] < 0.02) 
        terminate =  cage_points_mask.sum() > CAGE_POINT_THRESHOLD
        # maybe this is more robust (use_farthest_point)?
        cage_points_mask_reg = regularize_pc_point_count(cage_points_mask[:,None], 
                               CONFIG.uniform_num_pts, use_farthest_point=False)
        print('number of cage points %d' % cage_points_mask_reg.sum())
        terminate  = cage_points_mask_reg.sum() > CAGE_POINT_THRESHOLD
        return cage_points_mask, terminate


    def termination_heuristics(self, state):
        """
        Target depth heuristics for determining if grasp can be executed.
        The threshold is based on depth in the middle of the camera and the finger is near the bottom two sides
        用于确定是否可以执行抓取的目标深度启发法。
        阈值基于相机中部的深度，而手指靠近底部两侧。
        """

        point_state = state[0][0]
        target_mask = get_target_mask(point_state)  # 获取目标对象mask
        point_state = point_state[:3, target_mask].T
        depth_heuristics = self.graspnet_closure(point_state)  # 获取抓取质量--是否使用深度启发法
        if (depth_heuristics):
            print('object inside gripper? start retracting...')  # 抓手内有物体？开始收回...
        return depth_heuristics  # 返回深度启发法flag


    def preview_trajectory(self, state, remain_timestep, vis=False):
        """
        use the current point cloud to simulate observation and action for a trajectory
        this can be used to check trajectory before execution
        """
        print('in preview trajectory')
        state_origin = copy.deepcopy(state)
        sim_state = [state[0][0].copy(), state[0][1]] 

        joints = get_joints(self.joint_listener)  # 获取关节弧度数据
        ef_pose = get_ef_pose(self.pose_listener)
        ef_pose_origin = ef_pose.copy()
        joint_plan = [joints]
        ef_pose_plan = [ef_pose]

        for episode_steps in range(remain_timestep):
            state[0] = sim_state
            gaddpg_input_state = select_target_point(state)
            step = min(max(remain_timestep - episode_steps, 1), 25)
            action, _, _, aux_pred = agent.select_action(gaddpg_input_state, remain_timestep=step)
            action_pose = unpack_action(action)
            ef_pose = ef_pose.dot(action_pose)
            joints = solve_ik(joints, pack_pose(ef_pose))
            joint_plan.append(joints)
            ef_pose_plan.append(ef_pose)
            sim_next_point_state = se3_transform_pc(se3_inverse(action_pose), sim_state[0]) 
            sim_state[0] = sim_next_point_state

        if vis:
            # vis entire traj. Might be useful
            poses_ = robot.forward_kinematics_parallel(
                                wrap_value(joint_plan[0])[None], offset=True)[0]
            poses = [pack_pose(pose) for pose in poses_]
            line_starts, line_ends = grasp_gripper_lines(np.array(ef_pose_plan))
            points = state_origin[0][0]
            points = se3_transform_pc(ef_pose_origin, points)
            point_color = get_point_color(points)
            rgb = self.planner.planner_scene.renderer.vis(poses, list(range(10)), 
                shifted_pose=np.eye(4),
                interact=2,
                V=np.array(V),
                visualize_context={
                    "white_bg": True,
                    "project_point": [points],
                    "project_color": [point_color],
                    "static_buffer": True,
                    "reset_line_point": True,
                    "thickness": [2],
                    "line": [(line_starts[0], line_ends[0])],
                    "line_color": [[255, 0, 0]],            
                }
            )

        num = len(joint_plan)
        traj = np.zeros((num, 9), dtype=np.float32)
        for i in range(num):
            traj[i, :] = joint_plan[i]
        return traj


# for debuging
def send_transform(T, ef_pose, name, base_frame='measured/base_link'):
    broadcaster = tf.TransformBroadcaster()
    marker_pub = rospy.Publisher(name, Marker, queue_size = 10)
    for i in range(100):
        print('sending transformation {}'.format(name))
        qt = mat2quat(T[:3, :3])
        broadcaster.sendTransform(T[:3, 3], [qt[1], qt[2], qt[3], qt[0]], rospy.Time.now(), name, base_frame)

        GRASP_FRAME_OFFSET = tra.quaternion_matrix([0, 0, -0.707, 0.707])
        GRASP_FRAME_OFFSET[:3, 3] = [0, 0, 0.0]
        vis_pose = np.matmul(ef_pose, GRASP_FRAME_OFFSET)

        publish_grasps(marker_pub, base_frame, vis_pose)
        rospy.sleep(0.1)


def show_grasps(ef_poses, name, base_frame='measured/base_link'):
    marker_pub = rospy.Publisher(name, MarkerArray, queue_size = 10)
    GRASP_FRAME_OFFSET = tra.quaternion_matrix([0, 0, -0.707, 0.707])
    GRASP_FRAME_OFFSET[:3, 3] = [0, 0, 0.0]
    color = [0, 1, 0, 1]

    while not rospy.is_shutdown():
        markerArray = MarkerArray()
        for i in range(ef_poses.shape[0]):
            ef_pose = ef_poses[i]
            vis_pose = np.matmul(ef_pose, GRASP_FRAME_OFFSET)

            marker = create_gripper_marker_message (
                frame_id = base_frame,
                namespace = 'hand',
                mesh_resource = 'package://grasping_vae/panda_gripper.obj',
                color = color,
                marker_id = i,
            )
            pos = tra.translation_from_matrix(vis_pose)
            quat = tra.quaternion_from_matrix(vis_pose)
            marker.pose = Pose(position=Point(*pos), orientation=Quaternion(*quat))
            markerArray.markers.append(marker)

        # Renumber the marker IDs
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1

        marker_pub.publish(markerArray)
        print('publishing grasps')
        rospy.sleep(0.1)


def create_gripper_marker_message(
        frame_id, 
        namespace,
        mesh_resource, 
        color, 
        lifetime=True, 
        mesh_use_embedded_materials=True,                 
        marker_id=0, 
        frame_locked=False,):
    marker = Marker()
    marker.action = Marker.ADD
    marker.id = marker_id
    marker.ns = namespace
    if lifetime:
        marker.lifetime = rospy.Duration(0.2)
    marker.frame_locked = frame_locked
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.scale.x = marker.scale.y = marker.scale.z = 0.5
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = mesh_resource                                        
    marker.mesh_use_embedded_materials = mesh_use_embedded_materials            

    return marker


def publish_grasps(publisher, frame_id, grasp):
    color = [0, 1, 0, 1]
    marker = create_gripper_marker_message (
        frame_id=frame_id,
        namespace='hand',
        mesh_resource='package://grasping_vae/panda_gripper.obj',
        color=color,
        marker_id=0,
    )
    pos = tra.translation_from_matrix(grasp)
    quat = tra.quaternion_from_matrix(grasp)
    marker.pose = Pose(position=Point(*pos), orientation=Quaternion(*quat))
    publisher.publish(marker)



def make_pose(tf_pose):
    """
    Helper function to get a full matrix out of this pose
    从tf_pose这个姿势中获得完整的矩阵
    转换成4×4矩阵, pose[:3, :3] = orn (物体方向), pose[:3, 3] = trans (物体位置)
    """
    trans, rot = tf_pose  # trans位置，rot方向（四元数）
    pose = tra.quaternion_matrix(rot)  # 四元数-->3×3矩阵
    pose[:3, 3] = trans
    return pose


def gaddpg_grasps_from_simulate_view(gaddpg, state, time, ef_pose):
    """
    simulate views for gaddpg
    模拟DDPG的视图
    """
    n = 30
    mask = get_target_mask(state[0][0])
    point_state = state[0][0][:, mask]

    # hand to base
    point_state = se3_transform_pc(ef_pose, point_state)  # 坐标系转换
    print('target point shape', point_state.shape)
    # target center is in base coordinate now  目标中心现处于基准坐标
    target_center = point_state.mean(1)[:3]
    print('target center', target_center)

    # set up gaddpg  建立gaddpg
    img_state = state[0][1]
    gaddpg.policy.eval()  # 设置gaddpg策略网络为评估模式
    gaddpg.state_feature_extractor.eval()  # 设置gaddpg状态特征提取网络为评估模式

    # sample view (simulated hand) in base  底座中的示例视图（模拟手）
    view_poses = np.array(sample_ef_view_transform(n, 0.2, 0.5, target_center, linspace=True, anchor=True))

    # base to view (simulated hand)  基本视图（模拟手）
    inv_view_poses = se3_inverse_batch(view_poses)
    transform_view_points = np.matmul(inv_view_poses[:,:3,:3], point_state[:3]) + inv_view_poses[:,:3,[3]]
 
    # gaddpg generate grasps  gaddpg生成抓取
    point_state_batch = torch.from_numpy(transform_view_points).cuda().float()  # 将数组转换为张量
    time = torch.ones(len(point_state_batch) ).float().cuda() * 10. # time
    point_state_batch = torch.cat((point_state_batch, torch.zeros_like(point_state_batch)[:, [0]]), dim=1)
    policy_feat  = gaddpg.extract_feature(img_state, point_state_batch, value=False, time_batch=time)   # 提取特征
    _,_,_,gaddpg_aux  = gaddpg.policy.sample(policy_feat)  # gaddpg sample动作
 
    # compose with ef poses  组成ef姿态
    gaddpg_aux = gaddpg_aux.detach().cpu().numpy()
    unpacked_poses = [unpack_pose_rot_first(pose) for pose in gaddpg_aux]  # 将pose打包为4×4矩阵形式
    goal_pose_ws = np.matmul(view_poses, np.array(unpacked_poses)) # grasp to ef
    planner_cfg.external_grasps = goal_pose_ws

    # show_grasps(view_poses, 'grasps')
    # planner_cfg.external_grasps = view_poses
    # planner_cfg.external_grasps = np.concatenate((goal_pose_ws, view_poses), axis=0) # also visualize view
    planner_cfg.use_external_grasp = True  # 设置是否使用外部抓取为true


def select_target_point(state, target_pt_num=1024):
    """
    get target point cloud for gaddpg input
    获取gaddpg输入的目标点云
    """
    point_state = state[0][0]
    target_mask = get_target_mask(point_state)  # 获取目标点的mask
    # removing gripper point later  稍后移除抓手点
    point_state = point_state[:4, target_mask] # 获取目标点云  
    gripper_pc  = point_state[:4, :6] #  
    point_num  = min(point_state.shape[1], target_pt_num)  # 如果point_state.shape[1]小于1024
    obj_pc = regularize_pc_point_count(point_state.T, point_num, False).T  # 对目标点云进行采样
    point_state = np.concatenate((gripper_pc, obj_pc), axis=1)  # 合并矩阵
    return [(point_state, state[0][1])] + state[1:]  # 返回目标点云


def setup():
    """
    Set up networks with pretrained models and config as well as data migration
    """
    load_from_pretrain = args.pretrained is not None and os.path.exists(args.pretrained)

    if load_from_pretrain and not args.finetune:
        cfg_folder = args.pretrained
        cfg_from_file(os.path.join(cfg_folder, "config.yaml"), reset_model_spec=False)
        cfg.RL_MODEL_SPEC = os.path.join(cfg_folder, cfg.RL_MODEL_SPEC.split("/")[-1])
        dt_string = args.pretrained.split("/")[-1]
 
    else:
        if args.fix_output_time is None:
            dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        else:
            dt_string = args.fix_output_time

    model_output_dir = os.path.join(cfg.OUTPUT_DIR, dt_string)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    new_output_dir = not os.path.exists(model_output_dir) and not args.test
    print("Using config:")
    pprint.pprint(cfg)
    net_dict = make_nets_opts_schedulers(cfg.RL_MODEL_SPEC, cfg.RL_TRAIN)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    return net_dict, dt_string


def solve_ik(joints, pose):
    """
    For simulating trajectory
    """
    ik =  robot.inverse_kinematics(pose[:3], ros_quat(pose[3:]), seed=joints[:7])
    if ik is not None:
        joints = np.append(np.array(ik), [0.04, 0.04])
    return joints


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--env-name', default="PandaYCBEnv")
    parser.add_argument('--policy', default="DDPG" )
    parser.add_argument('--seed', type=int, default=123456, metavar='N' )
      
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--pretrained', type=str, default=None, help='test one model')
    parser.add_argument('--test', action="store_true", help='test one model')
    parser.add_argument('--log', action="store_true", help='log')
    parser.add_argument('--render', action="store_true", help='rendering')
    parser.add_argument('--record', action="store_true", help='record video')
    parser.add_argument('--test_episode_num', type=int, default=10, help='number of episodes to test')
    parser.add_argument('--finetune', action="store_true", help='deprecated')
    parser.add_argument('--expert', action="store_true", help='generate experte rollout')
    parser.add_argument('--num_runs',  type=int, default=1)
    parser.add_argument('--max_cnt_per_obj',  type=int, default=10)
    parser.add_argument('--model_surfix',  type=str, default='latest', help='surfix for loaded model')
    parser.add_argument('--rand_objs', action="store_true", help='random objects in Shapenet')
    parser.add_argument('--load_test_scene', action="store_true", help='load pregenerated random scenes')
    parser.add_argument('--change_dynamics', action="store_true", help='change dynamics of the object')
    parser.add_argument('--egl', action="store_true", help='use egl plugin in bullet')

    parser.add_argument('--config_file',  type=str, default=None)
    parser.add_argument('--output_file',  type=str, default='rollout_success.txt')
    parser.add_argument('--fix_output_time', type=str, default=None)
    parser.add_argument('--use_external_grasp', action="store_true")
    parser.add_argument('--vis_grasp_net', action="store_true")
    parser.add_argument('--start_idx',  type=int, default=1)
    parser.add_argument('--real_world', action="store_true")
    parser.add_argument('--preview_traj', action="store_true")
    parser.add_argument('--fix_initial_state', action="store_true")

    args = parser.parse_args()
    return args, parser


### TODO
def get_joints(joint_listener):
    """
    (9, ) robot joint in radians 
    just for rendering and simulating
    (9, ) 获取关节弧度数据
    仅为了渲染和模拟
    """      
    if LOCAL_TEST: # dummy  本地测试(伪造数据)
        return np.array([-0.5596, 0.5123, 0.5575, -1.6929, 0.2937, 1.6097, -1.237, 0.04, 0.04])
    else:
        joints = joint_listener.joint_position  # joint_listener向ROS Master询问关节弧度参数值
        print('robot joints', joints)  # 打印
        return joints

  
def get_ef_pose(pose_listener):
    """
    (4, 4) end effector pose matrix from base  (4, 4)末端执行器姿态矩阵（机械臂本体坐标系）
    """  
    if LOCAL_TEST: # dummy  本地测试(伪造数据)
        return np.array([[-0.1915,  0.8724, -0.4498,  0.6041],
                         [ 0.7355,  0.4309,  0.5228, -0.0031],
                         [ 0.6499, -0.2307, -0.7242,  0.3213],
                         [ 0.,      0.,      0.,      1.    ]])
    else:
        base_frame = 'measured/base_link'  # 机械臂本体坐标系
        target_frame = 'measured/panda_hand'  # 机械臂抓手坐标系
        try:
            tf_pose = pose_listener.lookupTransform(base_frame, target_frame, rospy.Time(0))  # 获得两个坐标系之间转换的关系，抓手坐标系-->本体坐标系
            pose = make_pose(tf_pose)  # 将tf_pose转换为pose 4×4矩阵
        except (tf2_ros.LookupException,  # 查找异常
            tf2_ros.ConnectivityException,  # 连接异常
            tf2_ros.ExtrapolationException):  # 推断异常 
            pose = None
            print('cannot find end-effector pose')
            sys.exit(1)  # 捕获这个异常，抛异常事件供捕获
        return pose


initial_joints = np.array([[-0.02535421982428639, -1.1120411124179306, 0.07915425984753728, -2.574433677700231, 0.0012470895074533914, 1.926161096378418, 0.9002216220876491],
                           [0.5805350207739269, -0.8111362388758844, -1.1146667134109263, -2.2735199081247064, -0.18589086490010281, 2.2351670468606946, -0.36534494081830765],
                           [-0.4345369377943954, -1.05069044781103, 1.119439285721959, -2.421638742837782, -0.02910207191286081, 2.0685257700621205, 1.5517931027048162],
                           [0.6299110230284048, -1.2067977417344766, -1.3116628687477672, -2.0905629379711166, -0.32998541843294193, 1.8464060782205653, -0.45038227560404887],
                           [-0.7665819353096028, -1.0393133004705655, 1.322218198802843, -2.0935060303990145, 0.33048455105753755, 1.8427947370070838, 1.746254150224718]])


if __name__ == '__main__':
    # Lirui: Replacing setup code
    # take a look at test_realworld for execution in ycb if necessary

    args, parser = parse_args()  # 获取参数

    print('Called with args:')
    print(args)

    # create robot 初始化节点(声明节点名字，只到rospy有这个信息，他才能开始和ROS Master通信)
    rospy.init_node("gaddpg")

    from OMG.ycb_render.robotPose import robot_pykdl
    robot = robot_pykdl.robot_kinematics(None, data_path='../../../')  # 用PyKDL构建机器人运动学，主要用于训练带有末端执行器的7自由度机器人手臂的视觉系统
 
    ############################# DEFINE RENDERER  定义renderer渲染器
    '''
    from OMG.ycb_render.ycb_renderer import YCBRenderer
    width, height = 640, 480
    renderer = YCBRenderer(width=width, height=height, offset=False)
    renderer.set_projection_matrix(width, height, width * 0.8, width * 0.8, width / 2, height / 2, 0.1, 6)
    renderer.set_camera_default()

    models = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "finger", "finger"]
    obj_paths = ["data/robots/{}.DAE".format(item) for item in models]
    renderer.load_objects(obj_paths)
    '''

    V =     [
                [-0.9351, 0.3518, 0.0428, 0.3037],
                [0.2065, 0.639, -0.741, 0.132],
                [-0.2881, -0.684, -0.6702, 1.8803],
                [0.0, 0.0, 0.0, 1.0],
            ]
    CAGE_POINT_THRESHOLD = 25

    ############################# SETUP MODEL  设定模型
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") 
    
    # 确保神经网络有相同的初始化
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # 是否自动加速
    torch.backends.cudnn.deterministic = True  # 为了确定算法，保证得到一样的结果

    net_dict, output_time = setup()
    CONFIG = cfg.RL_TRAIN
    LOCAL_TEST = False  # not using actual robot  不使用真正机器人

    # Args
    ## updated
    if GA_DDPG_ONLY:
        cfg.RL_MAX_STEP = 20
    else:
        cfg.RL_MAX_STEP = 50
        CONFIG.uniform_num_pts = 4096

    CONFIG.output_time = output_time
    CONFIG.off_policy = True   
    POLICY = 'DDPG' if CONFIG.RL else 'BC'    	
    CONFIG.index_file = 'ycb_large.json'

    # The default config?  默认配置？
    cfg.ROS_CAMERA = 'D415'
    cfg.SCALES_BASE = [1.0]

    # Metrics
    input_dim = CONFIG.feature_input_dim
    cnt = 0.
    object_performance = {}
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)
    pretrained_path = model_output_dir

    # graspnet
    graspnet_cfg = get_graspnet_config(parser)
    graspnet_cfg = joint_config(
        graspnet_cfg.vae_checkpoint_folder,
        graspnet_cfg.evaluator_checkpoint_folder,
    )        
    graspnet_cfg['threshold'] = 0.8
    graspnet_cfg['sample_based_improvement'] = False
    graspnet_cfg['num_refine_steps'] = 5  # 20 
    graspnet_cfg['num_samples'] = 200       

    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    g1 = tensorflow.compat.v1.Graph()
    with g1.as_default():
        sess = tensorflow.compat.v1.Session(config=config)
        with sess.as_default():
            grasp_estimator = GraspEstimator(graspnet_cfg)
            grasp_estimator.build_network()
            grasp_estimator.load_weights(sess)

    if CONTACT_GRASPNET:
        graspnet_cfg_contact = get_graspnet_config_contact()
        global_config = config_utils.load_config(graspnet_cfg_contact.ckpt_dir, batch_size=graspnet_cfg_contact.forward_passes, arg_configs=graspnet_cfg_contact.arg_configs)

        # Create a session
        g2 = tensorflow.compat.v1.Graph()
        config = tensorflow.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with g2.as_default():
            sess_contact = tensorflow.compat.v1.Session(config=config)
            with sess_contact.as_default():
                grasp_estimator_contact = GraspEstimatorContact(global_config)
                grasp_estimator_contact.build_network()
                saver = tensorflow.compat.v1.train.Saver(save_relative_paths=True)
                grasp_estimator_contact.load_weights(sess_contact, saver, graspnet_cfg_contact.ckpt_dir, mode='test')
    else:
        grasp_estimator_contact = None

    # GA-DDPG
    action_space = PandaTaskSpace6D()  
    agent = globals()[POLICY](input_dim, action_space, CONFIG) # 138  
    agent.setup_feature_extractor(net_dict, args.test)
    agent.load_model(pretrained_path, surfix=args.model_surfix, set_init_step=True)

    ############################# DEFINE ROS INTERFACE 
    listener = ImageListener(agent, grasp_estimator, grasp_estimator_contact)
    while not rospy.is_shutdown():  # rospy.is_shutdown()检查程序是否应该退出
       listener.run_network()
