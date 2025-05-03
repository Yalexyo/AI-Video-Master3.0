import os
import cv2
import json
import logging
import numpy as np
import torch
from torchvision.transforms import functional as F
from PIL import Image
from collections import Counter

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    """视频视觉分析器，检测和识别视频中的关键视觉元素"""
    
    # 视觉元素类型
    ELEMENT_TYPES = {
        "object": ["奶瓶", "婴儿", "妈妈", "喂食工具", "包装", "产品"],
        "scene": ["家庭", "户外", "厨房", "婴儿房", "客厅"],
        "action": ["喂奶", "抱婴儿", "冲奶", "展示产品"],
        "product": ["特写", "包装", "使用", "展示"]
    }
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        初始化视觉分析器
        
        Args:
            model_path: 模型路径，默认使用预训练模型
            confidence_threshold: 检测置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        
        logger.info("初始化视觉分析器")
        
        try:
            # 载入模型 - 这里使用通用的YOLOv5模型作为示例
            # 实际项目中可以用自定义的物体检测或场景识别模型替换
            if torch.cuda.is_available():
                logger.info("使用CUDA加速视觉分析")
                self.device = "cuda"
            else:
                logger.info("使用CPU进行视觉分析")
                self.device = "cpu"
                
            # 这里可以使用torch.hub加载预训练的YOLOv5模型
            # 或者加载本地保存的模型
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("视觉分析模型加载成功")
            
            # 场景分类模型 - 实际项目中可能需要单独一个模型
            # 这里可以使用ResNet等模型进行场景分类
            # self.scene_model = ...
            
            # 动作识别模型 - 实际项目中可能需要单独一个模型
            # self.action_model = ...
            
        except Exception as e:
            logger.error(f"视觉分析模型加载失败: {str(e)}")
            raise
    
    def analyze_frame(self, frame):
        """
        分析单帧图像
        
        Args:
            frame: 图像数据，OpenCV格式(BGR)或PIL图像
            
        Returns:
            dict: 分析结果，包含检测到的物体、场景等信息
        """
        if isinstance(frame, np.ndarray):
            # 将OpenCV格式(BGR)转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
        elif isinstance(frame, Image.Image):
            pil_image = frame
        else:
            logger.error(f"不支持的图像格式: {type(frame)}")
            return None
        
        try:
            # 使用模型进行预测
            results = self.model(pil_image)
            
            # 获取检测结果
            pred = results.pred[0].cpu().numpy()
            
            # 解析检测结果
            objects_detected = []
            for *box, conf, cls in pred:
                if conf >= self.confidence_threshold:
                    class_id = int(cls)
                    class_name = results.names[class_id]
                    objects_detected.append({
                        "class": class_name,
                        "confidence": float(conf),
                        "box": box
                    })
            
            # 场景分类 - 简化版，实际项目中可能需要专门的场景分类模型
            # 这里做一个简单的启发式判断
            scene_type = self._infer_scene_from_objects(objects_detected)
            
            # 动作识别 - 简化版
            action_type = self._infer_action_from_objects(objects_detected)
            
            # 产品展示类型判断
            product_display = self._infer_product_display(objects_detected, frame.shape)
            
            # 构建分析结果
            analysis_result = {
                "objects": objects_detected,
                "scene": scene_type,
                "action": action_type,
                "product_display": product_display
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"帧分析失败: {str(e)}")
            return None
    
    def _infer_scene_from_objects(self, objects):
        """
        根据检测到的物体推断场景类型
        这是一个简化版实现，实际项目中应使用专门的场景分类模型
        
        Args:
            objects: 检测到的物体列表
            
        Returns:
            str: 推断的场景类型
        """
        # 提取所有检测到的类别
        classes = [obj["class"] for obj in objects]
        
        # 简单规则推断
        if "sink" in classes or "refrigerator" in classes:
            return "厨房"
        elif "couch" in classes or "tv" in classes:
            return "客厅"
        elif "bed" in classes or "crib" in classes:
            return "婴儿房"
        elif "car" in classes or "tree" in classes or "bench" in classes:
            return "户外"
        else:
            return "家庭" if "person" in classes else "未知场景"
    
    def _infer_action_from_objects(self, objects):
        """
        根据检测到的物体推断动作类型
        这是一个简化版实现，实际项目中应使用动作识别模型
        
        Args:
            objects: 检测到的物体列表
            
        Returns:
            str: 推断的动作类型
        """
        # 提取所有检测到的类别
        classes = [obj["class"] for obj in objects]
        
        # 简单规则推断
        if "baby" in classes and "bottle" in classes:
            return "喂奶"
        elif "baby" in classes and "person" in classes:
            return "抱婴儿"
        elif "bottle" in classes:
            return "冲奶"
        elif any(item in classes for item in ["bottle", "packaging"]):
            return "展示产品"
        else:
            return "未知动作"
    
    def _infer_product_display(self, objects, frame_shape):
        """
        判断是否为产品展示特写
        
        Args:
            objects: 检测到的物体列表
            frame_shape: 帧的形状(高度,宽度)
            
        Returns:
            str: 产品展示类型
        """
        # 寻找产品相关物体
        product_objects = [obj for obj in objects if obj["class"] in ["bottle", "packaging", "box"]]
        
        if not product_objects:
            return "无产品展示"
            
        # 判断是否为特写
        for obj in product_objects:
            box = obj["box"]
            obj_width = box[2] - box[0]
            obj_height = box[3] - box[1]
            
            # 计算物体占据画面的比例
            width_ratio = obj_width / frame_shape[1]
            height_ratio = obj_height / frame_shape[0]
            
            # 如果物体占据画面较大比例，判定为特写
            if width_ratio > 0.3 or height_ratio > 0.3:
                return "产品特写"
        
        return "产品展示"
    
    def analyze_video(self, video_path, sampling_rate=1.5):
        """
        分析整个视频，每隔sampling_rate秒提取一帧进行分析
        
        Args:
            video_path: 视频文件路径
            sampling_rate: 采样率，每隔多少秒分析一帧
            
        Returns:
            dict: 视频分析时间线，包含每一帧的分析结果
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return None
                
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            logger.info(f"开始分析视频: {video_path}, 时长: {duration:.2f}秒, FPS: {fps}")
            
            # 确定采样帧
            sampling_frames = []
            current_time = 0
            
            while current_time < duration:
                frame_idx = int(current_time * fps)
                sampling_frames.append((frame_idx, current_time))
                current_time += sampling_rate
            
            # 分析采样帧
            timeline = []
            
            for frame_idx, timestamp in sampling_frames:
                # 设置当前帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # 读取当前帧
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"无法读取帧 {frame_idx}, 时间戳: {timestamp:.2f}秒")
                    continue
                
                # 分析当前帧
                result = self.analyze_frame(frame)
                
                if result:
                    # 添加时间戳信息
                    result["timestamp"] = timestamp
                    result["frame_idx"] = frame_idx
                    timeline.append(result)
            
            # 关闭视频文件
            cap.release()
            
            # 计算视频整体特征
            video_features = self._compute_video_features(timeline)
            
            # 构建最终结果
            video_analysis = {
                "video_path": video_path,
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "timeline": timeline,
                "features": video_features
            }
            
            logger.info(f"视频分析完成: {video_path}, 分析了 {len(timeline)} 帧")
            
            return video_analysis
            
        except Exception as e:
            logger.error(f"视频分析失败: {str(e)}")
            return None
    
    def _compute_video_features(self, timeline):
        """
        计算整个视频的特征统计
        
        Args:
            timeline: 时间线分析结果
            
        Returns:
            dict: 视频特征统计
        """
        if not timeline:
            return {}
            
        # 统计各类视觉元素出现频率
        objects_counter = Counter()
        scenes_counter = Counter()
        actions_counter = Counter()
        product_displays_counter = Counter()
        
        for frame in timeline:
            # 统计物体
            for obj in frame.get("objects", []):
                objects_counter[obj["class"]] += 1
            
            # 统计场景
            scenes_counter[frame.get("scene", "未知场景")] += 1
            
            # 统计动作
            actions_counter[frame.get("action", "未知动作")] += 1
            
            # 统计产品展示
            product_displays_counter[frame.get("product_display", "无产品展示")] += 1
        
        # 返回统计结果
        return {
            "dominant_objects": dict(objects_counter.most_common(5)),
            "dominant_scene": scenes_counter.most_common(1)[0][0] if scenes_counter else "未知场景",
            "dominant_action": actions_counter.most_common(1)[0][0] if actions_counter else "未知动作",
            "product_display_stats": dict(product_displays_counter),
            "frame_count": len(timeline)
        }
    
    def save_analysis(self, analysis, output_path):
        """
        保存视频分析结果到文件
        
        Args:
            analysis: 视频分析结果
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"视频分析结果已保存到 {output_path}")
        except Exception as e:
            logger.error(f"保存视频分析结果失败: {str(e)}") 