from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Union, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import json
from elasticsearch import Elasticsearch
import re
from functools import wraps
import logging
# 从vanna.qdrant导入Qdrant_VectorStore而不是ChromaDB相关类
from vanna.qdrant import Qdrant_VectorStore

from flask import jsonify, request
import time
import logging
import traceback
from typing import Dict, Any, List, Tuple

import logging
import time
import requests
from typing import List, Dict, Any, Optional

from typing import Any, List, Dict
import json

import threading
import queue
import time
import uuid
from typing import Dict, Callable, Any, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedVanna")

# 创建 OpenAI 客户端
client = OpenAI(
    api_key="sk-9f8124e18aa242af830c8a502c015c40",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义意图关键词全局字典
INTENT_KEYWORDS = {
    "status": ['状态', '取消', '完成', '支付', 'status', 'complete', 'cancel', '进度', '阶段'],
    "time": ['时间', '日期', '年', '月', '日', 'time', 'date', 'year', 'month', 'day', '周期', '期限'],
    "type": ['类型', '种类', '分类', 'type', 'category', 'kind', '型号', '系列'],
    "count": ['数量', '多少', '计数', 'count', 'how many', 'number of', '个数', '台数'],
    "aggregation": ['average', 'sum', 'total', 'count', 'mean', 'max', 'min', '平均', '总和', '最大', '最小', '统计'],
    "comparison": ['more than', 'less than', 'greater', 'smaller', 'between', 'compare', '超过', '小于', '大于', '比较',
                   '差异'],

    # 机械制造专业意图关键词
    "equipment": ['设备', '机器', '机械', '装置', '仪器', '工具', 'equipment', 'machine', 'machinery', 'device',
                  'tool'],
    "component": ['零件', '部件', '组件', '配件', '元件', 'component', 'part', 'assembly', 'module', '轴承', '齿轮',
                  '螺栓'],
    "material": ['材料', '原料', '物料', '金属', '合金', '钢材', '塑料', 'material', 'metal', 'alloy', 'steel',
                 'plastic'],
    "specification": ['规格', '参数', '尺寸', '公差', '精度', '直径', '长度', '宽度', '高度', 'specification',
                      'dimension', 'tolerance'],
    "process": ['工艺', '流程', '制造', '加工', '生产', '装配', '焊接', '铸造', 'process', 'manufacturing',
                'production', 'assembly'],
    "maintenance": ['维护', '保养', '检修', '维修', '保养', '润滑', 'maintenance', 'upkeep', 'repair', 'service',
                    'lubrication'],
    "troubleshooting": ['故障', '问题', '诊断', '排查', '修复', '异常', '排障', 'trouble', 'issue', 'fault', 'diagnose',
                        'error'],
    "quality": ['质量', '检验', '检测', '标准', '合格', 'quality', 'inspection', 'test', 'standard', 'qualified',
                '抽检'],
    "efficiency": ['效率', '产能', '产量', '输出', '运行', 'efficiency', 'productivity', 'output', 'rate', 'operation'],
    "cost": ['成本', '费用', '价格', '预算', '投入', 'cost', 'expense', 'price', 'budget', 'investment', '经济性'],
    "safety": ['安全', '防护', '事故', '危险', '风险', '保障', 'safety', 'protection', 'accident', 'hazard', 'risk'],
    "supply_chain": ['供应', '供应商', '采购', '库存', '物流', '交付', 'supply', 'vendor', 'procurement', 'inventory',
                     'logistics']
}


class AsyncTaskManager:
    """异步任务管理器，使用线程池处理耗时任务"""

    def __init__(self, max_workers=5):
        """
        初始化任务管理器

        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.workers = []
        self.results = {}  # 存储任务结果
        self.status = {}  # 存储任务状态 (pending, running, completed, failed)
        self.callbacks = {}  # 任务完成后的回调函数
        self.logger = logging.getLogger("AsyncTaskManager")
        self._start_workers()

    def _worker_loop(self):
        """工作线程循环，不断从队列获取任务并执行"""
        while True:
            try:
                # 从队列获取任务
                task_id, task_func, args, kwargs = self.task_queue.get()

                # 更新任务状态
                self.status[task_id] = "running"
                self.logger.info(f"开始执行任务 {task_id}")

                try:
                    # 执行任务
                    result = task_func(*args, **kwargs)
                    # 存储结果
                    self.results[task_id] = result
                    self.status[task_id] = "completed"
                    self.logger.info(f"任务 {task_id} 完成")

                    # 执行回调（如果有）
                    if task_id in self.callbacks and self.callbacks[task_id]:
                        try:
                            self.callbacks[task_id](result)
                            self.logger.info(f"任务 {task_id} 回调执行成功")
                        except Exception as e:
                            self.logger.error(f"任务 {task_id} 回调执行失败: {str(e)}")

                except Exception as e:
                    # 任务执行失败
                    self.status[task_id] = "failed"
                    self.results[task_id] = str(e)
                    self.logger.error(f"任务 {task_id} 执行失败: {str(e)}")

                # 标记任务完成
                self.task_queue.task_done()

            except Exception as e:
                self.logger.error(f"工作线程执行出错: {str(e)}")
                # 短暂休息以避免CPU占用过高
                time.sleep(0.1)

    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.workers.append(thread)
            self.logger.info(f"启动工作线程 {i + 1}")

    def submit_task(self, task_func: Callable, callback: Optional[Callable] = None, *args, **kwargs) -> str:
        """
        提交任务到队列

        Args:
            task_func: 要执行的函数
            callback: 任务完成后的回调函数，接收任务结果作为参数
            *args, **kwargs: 传递给任务函数的参数

        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        self.status[task_id] = "pending"

        if callback:
            self.callbacks[task_id] = callback

        # 将任务放入队列
        self.task_queue.put((task_id, task_func, args, kwargs))
        self.logger.info(f"提交任务 {task_id} 到队列")

        return task_id

    def get_task_status(self, task_id: str) -> Dict:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict: 包含任务状态和结果（如果已完成）
        """
        if task_id not in self.status:
            return {"status": "not_found"}

        result = {
            "status": self.status[task_id]
        }

        # 如果任务已完成或失败，包含结果
        if self.status[task_id] in ["completed", "failed"] and task_id in self.results:
            result["result"] = self.results[task_id]

        return result

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        等待任务完成并返回结果

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            Any: 任务结果

        Raises:
            TimeoutError: 如果等待超时
            ValueError: 如果任务不存在
            RuntimeError: 如果任务执行失败
        """
        if task_id not in self.status:
            raise ValueError(f"任务 {task_id} 不存在")

        start_time = time.time()
        while self.status[task_id] in ["pending", "running"]:
            time.sleep(0.1)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务 {task_id} 超时")

        if self.status[task_id] == "failed":
            raise RuntimeError(f"任务 {task_id} 执行失败: {self.results[task_id]}")

        return self.results[task_id]

    def clean_old_tasks(self, max_age: float = 3600):
        """清理超过max_age秒的旧任务"""
        current_time = time.time()
        to_remove = []

        for task_id, status_info in self.status.items():
            if status_info.get("timestamp", 0) < current_time - max_age:
                to_remove.append(task_id)

        for task_id in to_remove:
            self.results.pop(task_id, None)
            self.status.pop(task_id, None)
            self.callbacks.pop(task_id, None)


class DocumentChunker:
    def __init__(self, mechanical_terms_path="/dictionary/MechanicalWords.txt", max_chunk_size=1000, overlap=50,
                 model_name="m3e-base", doc_patterns=None, key_markers=None):
        """
        初始化文档分块器，使用语义模型辅助优化分块

        Args:
             mechanical_terms_path: 专业术语词典文件路径
            max_chunk_size: 最大块大小
            overlap: 块之间的重叠大小
            model_name: 使用的语义模型名称
            doc_patterns: 文档模式配置字典
            key_markers: 分块用的关键标记列表
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.terms_dict = self._load_mechanical_terms(mechanical_terms_path)

        # 设置文档模式配置
        self.doc_patterns = doc_patterns or {}

        # 设置关键标记
        self.key_markers = key_markers or [
            "业务场景:",
            "示例数据:",
            "业务规则:"
        ]

        # 初始化语义模型
        self.semantic_model = SentenceTransformer(
            model_name_or_path=f"/models/sentence-transformers_{model_name}",
            local_files_only=True
        )

    def _is_json(self, text: str) -> bool:
        """
        检测文本是否为有效的JSON

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否为有效的JSON
        """
        text = text.strip()
        # 快速检查：JSON必须以{ 或 [ 开头
        if not (text.startswith('{') or text.startswith('[')):
            return False

        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def _detect_json_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        在文本中检测JSON代码块

        Args:
            text: 源文本

        Returns:
            List[Tuple[int, int, str]]: 返回JSON块列表，每项包含(开始位置, 结束位置, JSON内容)
        """
        # 查找Markdown代码块中的JSON
        md_json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        md_matches = re.finditer(md_json_pattern, text, re.MULTILINE)

        blocks = []
        for match in md_matches:
            json_text = match.group(1).strip()
            if self._is_json(json_text):
                blocks.append((match.start(), match.end(), json_text))

        # 如果没有找到Markdown代码块，尝试寻找裸JSON
        if not blocks:
            # 这需要更谨慎，因为可能误报
            # 查找以{ 或 [ 开头，以} 或 ]结尾的大块文本
            raw_json_pattern = r'(\{[\s\S]*?\}|\[[\s\S]*?\])'
            for match in re.finditer(raw_json_pattern, text):
                json_text = match.group(1).strip()
                if self._is_json(json_text):
                    blocks.append((match.start(), match.end(), json_text))

        return blocks

    def _split_json_intelligently(self, json_text: str) -> List[str]:
        """
        智能分割JSON数据，保持每个分块的有效性

        Args:
            json_text: JSON文本

        Returns:
            List[str]: JSON分块列表
        """
        try:
            # 如果JSON小于最大块大小，直接返回
            if len(json_text) <= self.max_chunk_size:
                return [json_text]

            data = json.loads(json_text)
            chunks = []

            # 处理JSON数组
            if isinstance(data, list):
                return self._split_json_array(data)

            # 处理JSON对象
            elif isinstance(data, dict):
                return self._split_json_object(data)

            # 非预期的JSON类型，作为普通文本处理
            else:
                return [json_text]

        except json.JSONDecodeError:
            # JSON解析失败，作为普通文本处理
            return self._process_section_with_original_method(json_text)

    def _split_json_array(self, data: List) -> List[str]:
        """
        分割JSON数组，保持每个分块为有效的JSON数组

        Args:
            data: 要分割的JSON数组

        Returns:
            List[str]: JSON数组分块列表
        """
        chunks = []

        # 空数组直接返回
        if not data:
            return ['[]']

        # 计算每个元素的大概大小
        avg_element_size = len(json.dumps(data[0])) if data else 0

        # 计算每块的元素数量，确保不超过最大块大小
        batch_size = max(1, min(len(data), int(self.max_chunk_size / (avg_element_size + 5))))

        # 分批处理数组
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # 添加额外信息，标记这是第几批，共几批
            batch_info = {
                "_meta": {
                    "part": i // batch_size + 1,
                    "total_parts": (len(data) + batch_size - 1) // batch_size,
                    "elements": len(batch),
                    "total_elements": len(data)
                },
                "data": batch
            }
            chunk_json = json.dumps(batch_info, ensure_ascii=False, indent=2)
            chunks.append(chunk_json)

        return chunks

    def _split_json_object(self, data: Dict) -> List[str]:
        """
        分割大型JSON对象，按键进行分组

        Args:
            data: 要分割的JSON对象

        Returns:
            List[str]: JSON对象分块列表
        """
        # 如果是嵌套复杂对象，优先处理大的子对象/数组
        keys = list(data.keys())
        chunks = []

        # 计算每个键值对的大小
        key_sizes = {}
        for key in keys:
            key_sizes[key] = len(json.dumps({key: data[key]}))

        # 分组键：先处理大型嵌套结构
        large_keys = [k for k, size in key_sizes.items() if size > self.max_chunk_size / 2]
        small_keys = [k for k in keys if k not in large_keys]

        # 处理大型嵌套结构
        for key in large_keys:
            value = data[key]
            if isinstance(value, list):
                # 递归处理大型数组
                array_chunks = self._split_json_array(value)
                for i, chunk in enumerate(array_chunks):
                    chunks.append(json.dumps({
                        "key": key,
                        "part": i + 1,
                        "total_parts": len(array_chunks),
                        "value": json.loads(chunk)
                    }, ensure_ascii=False, indent=2))
            elif isinstance(value, dict):
                # 递归处理大型对象
                dict_chunks = self._split_json_object(value)
                for i, chunk in enumerate(dict_chunks):
                    chunks.append(json.dumps({
                        "key": key,
                        "part": i + 1,
                        "total_parts": len(dict_chunks),
                        "value": json.loads(chunk)
                    }, ensure_ascii=False, indent=2))
            else:
                # 单个大型值（如长文本）
                chunks.append(json.dumps({key: value}, ensure_ascii=False, indent=2))

        # 处理剩余的小型键值对
        current_chunk = {}
        current_size = 0

        for key in small_keys:
            item_json = json.dumps({key: data[key]})
            item_size = len(item_json)

            if current_size + item_size > self.max_chunk_size and current_chunk:
                # 当前块达到大小限制，保存并创建新块
                chunks.append(json.dumps(current_chunk, ensure_ascii=False, indent=2))
                current_chunk = {}
                current_size = 0

            # 添加到当前块
            current_chunk[key] = data[key]
            current_size += item_size

        # 保存最后一个块
        if current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False, indent=2))

        return chunks

    def _load_mechanical_terms(self, filepath):
        """加载专业术语词典 - 调整为适应一行英文一行中文的格式"""
        terms_dict = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines) - 1:
                    en_term = lines[i].strip()
                    zh_term = lines[i + 1].strip()
                    if en_term and zh_term:
                        terms_dict[en_term] = zh_term
                    i += 2
        except Exception as e:
            logger.warning(f"加载专业术语词典失败: {str(e)}")

        return terms_dict

    def _protect_terms(self, text):
        """保护专业术语不被分割"""
        protected_text = text
        for en_term, zh_term in self.terms_dict.items():
            if en_term in text:
                protected_text = protected_text.replace(en_term, f"[TERM_{en_term}_TERM]")
            if zh_term in text:
                protected_text = protected_text.replace(zh_term, f"[TERM_{zh_term}_TERM]")
        return protected_text

    def _restore_terms(self, text):
        """恢复专业术语"""
        restored_text = text
        for en_term, zh_term in self.terms_dict.items():
            restored_text = restored_text.replace(f"[TERM_{en_term}_TERM]", en_term)
            restored_text = restored_text.replace(f"[TERM_{zh_term}_TERM]", zh_term)
        return restored_text

    def is_semantic_boundary(self, left_text, right_text, threshold=0.6):
        """
        使用语义模型判断是否为合适的分块边界

        Args:
            left_text: 左侧文本
            right_text: 右侧文本
            threshold: 相似度阈值，低于此值视为合适的边界

        Returns:
            bool: 是否为合适的分块边界
        """
        # 边界检查：确保文本非空
        if not left_text or not right_text:
            return True

        # 取有限的上下文窗口
        left_context = left_text[-150:] if len(left_text) > 150 else left_text
        right_context = right_text[:150] if len(right_text) > 150 else right_text

        # 生成嵌入
        left_emb = self.semantic_model.encode(left_context, convert_to_numpy=True)
        right_emb = self.semantic_model.encode(right_context, convert_to_numpy=True)

        # 计算余弦相似度
        similarity = np.dot(left_emb, right_emb) / (np.linalg.norm(left_emb) * np.linalg.norm(right_emb))

        # 相似度低于阈值，认为是合适的分块边界
        return similarity < threshold

    def chunk_document(self, document):
        """
        主分块方法，结合固定格式特征和语义边界检测，优化处理大型表格
        """
        # 边界检查：空文档处理
        if not document or not document.strip():
            return []

        # 边界检查：如果文档小于最大块大小，直接返回
        if len(document) <= self.max_chunk_size:
            return [document]

        # 保护专业术语
        protected_text = self._protect_terms(document)

        # 1. 检测是否包含JSON代码块
        json_blocks = self._detect_json_blocks(protected_text)

        if json_blocks:
            result_chunks = []
            last_position = 0

            # 处理交替的文本和JSON块
            for start, end, json_text in json_blocks:
                # 处理JSON块前的文本
                if start > last_position:
                    before_text = protected_text[last_position:start]
                    if before_text.strip():
                        # 使用原有方法处理JSON块之前的文本
                        before_chunks = self._process_section_with_original_method(before_text)
                        result_chunks.extend(before_chunks)

                # 处理JSON块
                json_chunks = self._split_json_intelligently(json_text)

                # 添加适当的上下文，确保模型理解这是分块的JSON
                for i, chunk in enumerate(json_chunks):
                    # 为JSON添加分块信息
                    context = f"## JSON数据（第{i + 1}部分，共{len(json_chunks)}部分）\n\n```json\n{chunk}\n```"
                    result_chunks.append(context)

                last_position = end

            # 处理最后一个JSON块后的文本
            if last_position < len(protected_text):
                after_text = protected_text[last_position:]
                if after_text.strip():
                    after_chunks = self._process_section_with_original_method(after_text)
                    result_chunks.extend(after_chunks)

            # 恢复专业术语并返回
            return [self._restore_terms(chunk) for chunk in result_chunks]

        # 2. 首先尝试按表定义分块
        table_sections = []
        # 使用配置的表定义模式
        table_pattern = self.doc_patterns.get("table_definition", {}).get(
            "block_pattern", r'(# 表名词:.+?)(?=# 表名词:|$)'
        )
        table_matches = re.finditer(table_pattern, protected_text, re.DOTALL)

        for match in table_matches:
            table_sections.append(match.group(1))

        # 如果按表分块成功，优先处理表级别分块
        if table_sections:
            result_chunks = []

            for section in table_sections:
                # 检测是否包含表格
                table_start = section.find("\n|")

                if table_start > 0:
                    # 检查表格大小
                    lines = section.split("\n")
                    header_line_idx = -1
                    separator_line_idx = -1

                    # 查找表头行和分隔行
                    for i, line in enumerate(lines):
                        if line.strip().startswith("|") and header_line_idx == -1:
                            header_line_idx = i
                        elif line.strip().startswith("|") and header_line_idx != -1 and "-" in line:
                            # 分隔行通常含有连字符
                            separator_line_idx = i
                            break

                    # 计算列数
                    if header_line_idx != -1:
                        header_line = lines[header_line_idx]
                        column_count = header_line.count("|") - 1

                        # 大表格处理策略(超过20列视为大表格)
                        if column_count > 20:
                            # 提取表前内容
                            before_table = "\n".join(lines[:header_line_idx])
                            if before_table:
                                result_chunks.append(before_table)

                            # 获取表头部分
                            table_header = "\n".join(
                                lines[header_line_idx:separator_line_idx + 1]) if separator_line_idx != -1 else lines[
                                header_line_idx]

                            # 计算表格主体范围
                            table_body_start = separator_line_idx + 1 if separator_line_idx != -1 else header_line_idx + 1
                            table_body_end = table_body_start

                            # 查找表格结束位置
                            for i in range(table_body_start, len(lines)):
                                if lines[i].strip().startswith("|"):
                                    table_body_end = i
                                elif table_body_end >= table_body_start and lines[i].strip() and not lines[
                                    i].strip().startswith("|"):
                                    break

                            # 获取表格数据行
                            data_rows = lines[table_body_start:table_body_end + 1]

                            # 表格分块：每个块含表头+少量数据行
                            header_context = f"## 示例数据\n\n{table_header}\n"
                            max_rows_per_chunk = 5  # 每块最多5行数据

                            for i in range(0, len(data_rows), max_rows_per_chunk):
                                chunk_rows = data_rows[i:i + max_rows_per_chunk]
                                table_chunk = header_context + "\n".join(chunk_rows)
                                result_chunks.append(table_chunk)

                            # 处理表格后内容
                            if table_body_end + 1 < len(lines):
                                after_table = "\n".join(lines[table_body_end + 1:])
                                if len(after_table) <= self.max_chunk_size:
                                    result_chunks.append(after_table)
                                else:
                                    # 继续使用原有的分块方法处理剩余内容
                                    after_chunks = self._process_section_with_original_method(after_table)
                                    result_chunks.extend(after_chunks)
                        else:
                            # 小表格处理，尝试保持完整
                            if len(section) <= self.max_chunk_size:
                                result_chunks.append(section)
                            else:
                                # 回退到原有分块逻辑
                                section_chunks = self._process_section_with_original_method(section)
                                result_chunks.extend(section_chunks)
                    else:
                        # 没有找到表头，使用原有分块逻辑
                        section_chunks = self._process_section_with_original_method(section)
                        result_chunks.extend(section_chunks)
                else:
                    # 没有表格，使用原有分块逻辑
                    section_chunks = self._process_section_with_original_method(section)
                    result_chunks.extend(section_chunks)

            # 恢复专业术语并返回
            return [self._restore_terms(chunk) for chunk in result_chunks]

        # 3. 如果不是按表结构组织的，回退到原有的基于标记的分块方法
        # 找出所有关键标记的位置 JSON数据和表格数据不可放在下面，避免重复向量化数据
        marker_positions = []
        for marker in self.key_markers:
            for match in re.finditer(re.escape(marker), protected_text):
                marker_positions.append(match.start())

        # 加入文本开始和结束位置
        marker_positions = [0] + sorted(marker_positions) + [len(protected_text)]

        for marker in key_markers:
            for match in re.finditer(re.escape(marker), protected_text):
                marker_positions.append(match.start())

        # 加入文本开始和结束位置
        marker_positions = [0] + sorted(marker_positions) + [len(protected_text)]

        # 生成初步分块
        initial_chunks = []
        for i in range(len(marker_positions) - 1):
            start = marker_positions[i]
            end = marker_positions[i + 1]

            chunk_text = protected_text[start:end]
            if chunk_text.strip():  # 忽略空块
                initial_chunks.append(chunk_text)

        # 4. 对超出最大长度的块进行进一步分割
        result_chunks = []
        for chunk in initial_chunks:
            if len(chunk) <= self.max_chunk_size:
                result_chunks.append(chunk)
            else:
                # 按段落分割
                paragraphs = re.split(r'\n\s*\n', chunk)
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n"
                        current_chunk += para
                    else:
                        # 当前段落会导致超出最大长度
                        if current_chunk:
                            # 检查语义边界
                            window_size = min(100, len(current_chunk), len(para))
                            if self.is_semantic_boundary(
                                    current_chunk[-window_size:],
                                    para[:window_size]
                            ):
                                # 是语义边界，直接分块
                                result_chunks.append(current_chunk)
                                current_chunk = para
                            else:
                                # 不是理想的语义边界，但必须分块
                                result_chunks.append(current_chunk)
                                current_chunk = para
                        else:
                            # 单个段落太长，需要强制分割
                            if len(para) > self.max_chunk_size:
                                sub_chunks = self._split_long_paragraph(para)
                                result_chunks.extend(sub_chunks)
                            else:
                                current_chunk = para

                if current_chunk:
                    result_chunks.append(current_chunk)

        # 5. 确保相邻块有适当的重叠
        final_chunks = []
        for i, chunk in enumerate(result_chunks):
            if i > 0 and self.overlap > 0:
                # 添加上一个块的末尾作为重叠
                prev_end = result_chunks[i - 1][-self.overlap:] if len(result_chunks[i - 1]) > self.overlap else \
                    result_chunks[i - 1]
                if not chunk.startswith(prev_end):
                    chunk = prev_end + chunk
            final_chunks.append(chunk)

        # 6. 恢复专业术语并返回
        return [self._restore_terms(chunk) for chunk in final_chunks]

    def _process_section_with_original_method(self, section):
        """使用原有分块方法处理部分内容"""
        if len(section) <= self.max_chunk_size:
            return [section]

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', section)
        current_chunk = ""
        result_chunks = []

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                # 检查语义边界
                if current_chunk:
                    window_size = min(100, len(current_chunk), len(para))
                    if self.is_semantic_boundary(
                            current_chunk[-window_size:],
                            para[:window_size]
                    ):
                        # 是语义边界，直接分块
                        result_chunks.append(current_chunk)
                        current_chunk = para
                    else:
                        # 不是理想的语义边界，但必须分块
                        result_chunks.append(current_chunk)
                        current_chunk = para
                else:
                    # 单个段落太长，需要强制分割
                    if len(para) > self.max_chunk_size:
                        sub_chunks = self._split_long_paragraph(para)
                        result_chunks.extend(sub_chunks)
                    else:
                        current_chunk = para

        if current_chunk:
            result_chunks.append(current_chunk)

        return result_chunks

    def _split_long_paragraph(self, paragraph):
        """处理超长段落，使用语义边界判断"""
        # 边界检查
        if len(paragraph) <= self.max_chunk_size:
            return [paragraph]

        chunks = []
        remaining = paragraph

        # 防止无限循环
        safety_counter = 0
        max_iterations = 100

        while len(remaining) > self.max_chunk_size and safety_counter < max_iterations:
            safety_counter += 1

            # 找一个接近最大长度的句子边界
            potential_end = self.max_chunk_size

            # 尝试在附近找句号等标点
            sentence_endings = [m.start() for m in re.finditer(r'[。！？!?]', remaining[:self.max_chunk_size + 100])]
            if sentence_endings:
                closest_ending = min(sentence_endings, key=lambda x: abs(x - self.max_chunk_size))
                if closest_ending > 0:
                    potential_end = closest_ending + 1  # 包括标点

            # 检查语义边界
            if potential_end < len(remaining) - 50:
                left = remaining[:potential_end]
                right = remaining[potential_end:potential_end + 100]

                if self.is_semantic_boundary(left, right):
                    chunks.append(remaining[:potential_end])
                    remaining = remaining[potential_end:]
                else:
                    # 不理想但必须分割
                    chunks.append(remaining[:potential_end])
                    remaining = remaining[potential_end:]
            else:
                # 剩余部分不长，全部加入
                chunks.append(remaining)
                break

        # 添加剩余部分
        if remaining and safety_counter < max_iterations:
            chunks.append(remaining)

        return chunks

# 修正utils类中的serialize_data函数
class utils:
    @staticmethod
    def serialize_data(data):
        """序列化数据，确保JSON兼容"""
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        if isinstance(data, list):
            return [utils.serialize_data(item) for item in data]

        if isinstance(data, dict):
            return {str(k): utils.serialize_data(v) for k, v in data.items()}

        # 尝试转换为字符串
        try:
            return str(data)
        except:
            return "不可序列化的值"


class RetrievalService:
    """封装检索和重排序逻辑的服务类"""

    def __init__(self, vanna_instance, config=None):
        """
        初始化检索服务

        Args:
            vanna_instance: Vanna实例，用于执行检索操作
            config: 配置字典
        """
        self.vn = vanna_instance
        self.config = config or {}

        # 配置重排序服务
        self.rerank_url = self.config.get("rerank_url", "http://localhost:8091")
        self.rerank_enabled = self.config.get("rerank_enabled", True)
        self.rerank_timeout = self.config.get("rerank_timeout", 120)

        # 检索配置
        self.max_results = self.config.get("max_results", 10)

        logger.info(f"检索服务初始化完成, 重排序{'启用' if self.rerank_enabled else '禁用'}")

    def retrieve(self, question: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute comprehensive retrieval pipeline with intent support

        Args:
            question: User question
            options: Retrieval options including:
                - max_results: Maximum results per type
                - enhance_query: Whether to enhance the query
                - use_rerank: Whether to use reranking
                - intent_type: Query intent type

        Returns:
            Dict containing retrieval results and statistics
        """
        start_time = time.time()
        options = options or {}
        stats = {}

        # 获取选项
        max_results = options.get("max_results", self.max_results)
        enhance_query = options.get("enhance_query", True)
        use_rerank = options.get("use_rerank", self.rerank_enabled)
        intent_type = options.get("intent_type")  # 获取意图类型

        # 准备检索参数
        retrieval_kwargs = {
            'rerank': use_rerank,
            'intent_type': intent_type  # 传递意图类型
        }

        # 增强查询（如果需要）
        if enhance_query and hasattr(self.vn, "preprocess_field_names"):
            enhanced_question = self.vn.preprocess_field_names(question)
            stats["enhanced_question"] = enhanced_question

        # 执行SQL示例检索
        sql_start = time.time()
        question_sql_list = self.vn.get_similar_question_sql(question, **retrieval_kwargs)
        stats["sql_retrieval_time"] = round(time.time() - sql_start, 3)
        stats["sql_results_count"] = len(question_sql_list)

        # 执行DDL检索
        ddl_start = time.time()
        ddl_list = self.vn.get_related_ddl(question, **retrieval_kwargs)
        stats["ddl_retrieval_time"] = round(time.time() - ddl_start, 3)
        stats["ddl_results_count"] = len(ddl_list)

        # 执行文档检索
        doc_start = time.time()
        doc_list = self.vn.get_related_documentation(question, **retrieval_kwargs)
        stats["doc_retrieval_time"] = round(time.time() - doc_start, 3)
        stats["doc_results_count"] = len(doc_list)

        # 提取表关系
        table_relationships = ""
        if hasattr(self.vn, "_extract_table_relationships"):
            table_relationships = self.vn._extract_table_relationships(ddl_list)

        # 限制结果数量
        if max_results > 0:
            if len(question_sql_list) > max_results:
                question_sql_list = question_sql_list[:max_results]
            if len(ddl_list) > max_results:
                ddl_list = ddl_list[:max_results]
            if len(doc_list) > max_results:
                doc_list = doc_list[:max_results]

        # 返回结果
        results = {
            "question_sql_list": question_sql_list,
            "ddl_list": ddl_list,
            "doc_list": doc_list,
            "table_relationships": table_relationships,
            "stats": stats,
            "total_time": round(time.time() - start_time, 3),
            "detected_intent": intent_type  # 包含检测到的意图
        }

        return results

# 向量和BM25结果融合算法
class RankFusion:
    @staticmethod
    def reciprocal_rank_fusion(results_lists: List[List[Any]], weights: List[float] = None, k: int = 60) -> List[Any]:
        """
        使用加权互惠排序融合（RRF）将多个结果列表融合。

        Args:
            results_lists: 结果列表的列表，每个列表按相关性排序
            weights: 各个结果列表的权重，默认权重相等
            k: RRF常数，控制排名的影响

        Returns:
            融合后的结果列表
        """
        if weights is None:
            weights = [1.0] * len(results_lists)

        if len(weights) != len(results_lists):
            raise ValueError("权重列表长度必须与结果列表数量相同")

        scores = {}

        # 为每个结果列表计算RRF分数
        for list_idx, (result_list, weight) in enumerate(zip(results_lists, weights)):
            for rank, item in enumerate(result_list, start=1):
                # 对JSON对象，转为字符串作为键
                if isinstance(item, dict):
                    item_key = json.dumps(item, sort_keys=True)
                else:
                    item_key = str(item)

                # 初始化该项的记录
                if item_key not in scores:
                    scores[item_key] = {"item": item, "score": 0.0, "sources": set()}

                # 计算加权RRF分数并累加
                scores[item_key]["score"] += weight * (1.0 / (k + rank))
                scores[item_key]["sources"].add(list_idx)

        # 上下文奖励：出现在多个来源的结果获得额外奖励
        for item_key, data in scores.items():
            # 来源多样性奖励 - 同时出现在多个搜索结果中的项目得分提升
            source_diversity = len(data["sources"]) / len(results_lists)
            data["score"] *= (1 + 0.2 * source_diversity)  # 最多20%的奖励

        # 按分数降序排序
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]

    @staticmethod
    def multi_stage_fusion(query: str, search_functions: Dict[str, Tuple[callable, float]], k: int = 60) -> List[Any]:
        """
        多阶段融合算法，针对相近语义字段进行优化

        Args:
            query: 原始查询
            search_functions: 搜索函数字典，格式为 {名称: (搜索函数, 权重)}
            k: RRF常数

        Returns:
            融合后的结果列表
        """
        # 第一阶段: 直接检索
        stage1_results = {}
        stage1_lists = []
        stage1_weights = []

        for name, (search_fn, weight) in search_functions.items():
            results = search_fn(query)
            stage1_results[name] = results
            stage1_lists.append(results)
            stage1_weights.append(weight)

        # 第二阶段: 提取关键词
        key_terms = RankFusion._extract_key_terms(query, stage1_results)

        # 第三阶段: 使用关键词增强查询进行二次检索
        stage2_results = {}
        for term in key_terms:
            enhanced_query = f"{query} {term}"
            term_results = []

            for name, (search_fn, _) in search_functions.items():
                results = search_fn(enhanced_query)
                term_results.extend(results)

            stage2_results[term] = term_results

        # 第四阶段: 融合所有结果
        all_candidates = {}

        # 添加第一阶段结果
        for name, results in stage1_results.items():
            weight = search_functions[name][1]
            for rank, item in enumerate(results, start=1):
                item_key = RankFusion._get_item_key(item)

                if item_key not in all_candidates:
                    all_candidates[item_key] = {"item": item, "score": 0, "sources": set()}

                all_candidates[item_key]["score"] += weight * (1.0 / (k + rank))
                all_candidates[item_key]["sources"].add(f"direct_{name}")

        # 添加第二阶段结果 (关键词增强)
        term_weight = 0.3 / len(stage2_results) if stage2_results else 0
        for term, results in stage2_results.items():
            for rank, item in enumerate(results, start=1):
                item_key = RankFusion._get_item_key(item)

                if item_key not in all_candidates:
                    all_candidates[item_key] = {"item": item, "score": 0, "sources": set()}

                all_candidates[item_key]["score"] += term_weight * (1.0 / (k + rank))
                all_candidates[item_key]["sources"].add(f"term_{term}")

        # 多样性奖励
        for item_key, data in all_candidates.items():
            # 基于来源数量的多样性奖励
            source_count = len(data["sources"])
            direct_sources = sum(1 for s in data["sources"] if s.startswith("direct_"))
            term_sources = sum(1 for s in data["sources"] if s.startswith("term_"))

            # 混合来源的结果获得更高奖励
            if direct_sources > 0 and term_sources > 0:
                data["score"] *= 1.3  # 30%的奖励

            # 来源数量越多，奖励越高
            data["score"] *= (1 + 0.1 * min(5, source_count) / 5)

        # 按分数排序
        sorted_items = sorted(all_candidates.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]

    @staticmethod
    def _extract_key_terms(query: str, results: Dict[str, List]) -> List[str]:
        """从查询和结果中提取可能的关键词"""
        terms = []

        # 1. 从查询中提取潜在字段名和值
        # 匹配可能的字段名
        field_patterns = [
            (r'([a-z]+[A-Z][a-zA-Z]*)', 1),  # 驼峰命名
            (r'([a-z_]+)', 1),  # 下划线命名
            (r'"([^"]+)"', 1),  # 双引号包围
            (r"'([^']+)'", 1)  # 单引号包围
        ]

        for pattern, group in field_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m and len(m) > 2:  # 忽略太短的词
                            terms.append(m)
                elif match and len(match) > 2:
                    terms.append(match)

        # 2. 提取特殊业务术语
        business_terms = {
            "订单": ["order", "purchase"],
            "客户": ["customer", "client", "user"],
            "产品": ["product", "item", "goods"],
            "支付": ["payment", "transaction"],
            "状态": ["status", "state", "condition"],
            "取消": ["cancel", "cancelled", "canceled", "cancellation"],
            "完成": ["complete", "completed", "finish", "finished"],
            "收货": ["delivery", "delivered", "receive", "received"]
        }

        for term, synonyms in business_terms.items():
            if term in query:
                terms.extend(synonyms)

        # 3. 转换驼峰/下划线变体
        variants = []
        for term in terms:
            # 驼峰转下划线
            if any(c.isupper() for c in term):
                snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in term]).lstrip('_')
                variants.append(snake_case)

            # 下划线转驼峰
            if '_' in term:
                camel_case = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(term.split('_')))
                variants.append(camel_case)

        terms.extend(variants)

        # 4. 从结果中提取高频词
        # 这里可以实现更复杂的逻辑

        # 去重并返回
        return list(set(terms))

    @staticmethod
    def _get_item_key(item) -> str:
        """获取项目的唯一键"""
        if isinstance(item, dict):
            return json.dumps(item, sort_keys=True)
        return str(item)

    @staticmethod
    def contextual_fusion(query: str, dense_results: list, lexical_results: list, metadata=None, k: int = 60,
                          rerank_service_url: str = None, intent_type: str = None) -> list:
        """
        Context-aware fusion algorithm with intent detection and integrated reranking

        Args:
            query: User query
            dense_results: Vector search results (or reranked results)
            lexical_results: Text search results
            metadata: Metadata information
            k: RRF constant
            rerank_service_url: Optional URL for reranking service
            intent_type: Query intent type from upstream analysis (optional)

        Returns:
            Fused result list
        """
        # 默认权重
        vector_weight = 0.7
        lexical_weight = 0.3

        # 如果没有提供intent_type，尝试检测意图
        if not intent_type:
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))

            # 检测通用意图类型
            is_status_query = any(term in query_terms for term in INTENT_KEYWORDS["status"])
            is_time_query = any(term in query_terms for term in INTENT_KEYWORDS["time"])
            is_type_query = any(term in query_terms for term in INTENT_KEYWORDS["type"])
            is_count_query = any(term in query_terms for term in INTENT_KEYWORDS["count"])
            is_aggregation_query = any(term in query.lower() for term in INTENT_KEYWORDS["aggregation"])
            is_comparison_query = any(term in query.lower() for term in INTENT_KEYWORDS["comparison"])

            # 检测机械制造专业意图
            is_equipment_query = any(term in query_terms for term in INTENT_KEYWORDS["equipment"])
            is_component_query = any(term in query_terms for term in INTENT_KEYWORDS["component"])
            is_material_query = any(term in query_terms for term in INTENT_KEYWORDS["material"])
            is_specification_query = any(term in query_terms for term in INTENT_KEYWORDS["specification"])
            is_process_query = any(term in query_terms for term in INTENT_KEYWORDS["process"])
            is_maintenance_query = any(term in query_terms for term in INTENT_KEYWORDS["maintenance"])
            is_troubleshooting_query = any(term in query_terms for term in INTENT_KEYWORDS["troubleshooting"])
            is_quality_query = any(term in query_terms for term in INTENT_KEYWORDS["quality"])
            is_efficiency_query = any(term in query_terms for term in INTENT_KEYWORDS["efficiency"])
            is_cost_query = any(term in query_terms for term in INTENT_KEYWORDS["cost"])
            is_safety_query = any(term in query_terms for term in INTENT_KEYWORDS["safety"])
            is_supply_chain_query = any(term in query_terms for term in INTENT_KEYWORDS["supply_chain"])

            # 确定主要意图
            if is_status_query:
                intent_type = "status"
            elif is_time_query:
                intent_type = "time"
            elif is_type_query:
                intent_type = "type"
            elif is_count_query or is_aggregation_query:
                intent_type = "count" if is_count_query else "aggregation"
            elif is_comparison_query:
                intent_type = "comparison"
            elif is_equipment_query:
                intent_type = "equipment"
            elif is_component_query:
                intent_type = "component"
            elif is_material_query:
                intent_type = "material"
            elif is_specification_query:
                intent_type = "specification"
            elif is_process_query:
                intent_type = "process"
            elif is_maintenance_query:
                intent_type = "maintenance"
            elif is_troubleshooting_query:
                intent_type = "troubleshooting"
            elif is_quality_query:
                intent_type = "quality"
            elif is_efficiency_query:
                intent_type = "efficiency"
            elif is_cost_query:
                intent_type = "cost"
            elif is_safety_query:
                intent_type = "safety"
            elif is_supply_chain_query:
                intent_type = "supply_chain"
            else:
                intent_type = "general"

        # 基于意图类型设置权重
        if intent_type == "status" or intent_type == "type":
            # 状态和类型查询，BM25可能更准确
            vector_weight = 0.4
            lexical_weight = 0.6
        elif intent_type == "time":
            # 时间查询，两者都重要
            vector_weight = 0.5
            lexical_weight = 0.5
        elif intent_type in ["count", "aggregation"]:
            # 计数/聚合查询通常需要精确匹配
            vector_weight = 0.35
            lexical_weight = 0.65
        elif intent_type == "comparison":
            # 比较查询需要语义理解
            vector_weight = 0.6
            lexical_weight = 0.4
        elif intent_type == "specification":
            # 规格参数查询通常需要精确匹配
            vector_weight = 0.3
            lexical_weight = 0.7
        elif intent_type == "troubleshooting":
            # 故障诊断查询受益于语义相似性
            vector_weight = 0.65
            lexical_weight = 0.35
        elif intent_type in ["material", "component"]:
            # 材料和零部件查询需要精确匹配
            vector_weight = 0.45
            lexical_weight = 0.55
        elif intent_type == "safety":
            # 安全规范查询需要高度准确性
            vector_weight = 0.35
            lexical_weight = 0.65
        elif intent_type == "process":
            # 工艺流程查询需要语义理解
            vector_weight = 0.6
            lexical_weight = 0.4
        else:
            # 默认权重
            vector_weight = 0.7
            lexical_weight = 0.3

        # 应用外部重排序(如果提供URL)
        if rerank_service_url:
            try:
                # 合并所有候选项用于重排序
                all_candidates = []
                seen_items = set()

                # 处理dense_results
                for item in dense_results:
                    item_key = RankFusion._get_item_key(item)
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        all_candidates.append({"content": str(item), "item": item})

                # 添加lexical_results
                for item in lexical_results:
                    item_key = RankFusion._get_item_key(item)
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        all_candidates.append({"content": str(item), "item": item})

                # 调用重排序服务
                response = requests.post(
                    rerank_service_url,
                    json={
                        "query": query,
                        "documents": [{"content": c["content"]} for c in all_candidates],
                        "top_k": len(all_candidates),
                        "intent_type": intent_type  # 传递意图类型给重排序服务
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    # 处理重排序结果
                    reranked_data = response.json()
                    reranked_results = []

                    # 按新顺序重建原始列表
                    for item in reranked_data.get("results", []):
                        idx = item.get("index")
                        if 0 <= idx < len(all_candidates):
                            reranked_results.append(all_candidates[idx]["item"])

                    # 直接返回重排序结果
                    return reranked_results
            except Exception as e:
                logger.error(f"Error during reranking in contextual_fusion: {str(e)}", exc_info=True)

        # 计算融合分数
        scores = {}

        # 处理向量结果
        for rank, item in enumerate(dense_results, start=1):
            item_key = RankFusion._get_item_key(item)
            if item_key not in scores:
                scores[item_key] = {"item": item, "score": 0, "matches": set()}

            # 检查item是否有rerank_score
            if isinstance(item, dict) and "rerank_score" in item:
                # 使用rerank_score作为提升
                rerank_boost = min(1.5, 1.0 + item["rerank_score"] / 2)  # 最高50%提升
                scores[item_key]["score"] += vector_weight * (1.0 / (k + rank)) * rerank_boost
            else:
                scores[item_key]["score"] += vector_weight * (1.0 / (k + rank))

            scores[item_key]["matches"].add("vector")

        # 处理文本结果
        for rank, item in enumerate(lexical_results, start=1):
            item_key = RankFusion._get_item_key(item)
            if item_key not in scores:
                scores[item_key] = {"item": item, "score": 0, "matches": set()}

            scores[item_key]["score"] += lexical_weight * (1.0 / (k + rank))
            scores[item_key]["matches"].add("lexical")

            # 计算精确词汇匹配奖励
            item_str = str(item).lower()

            # 提取查询词
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))

            # 计算查询词匹配率
            term_matches = sum(1 for term in query_terms if term in item_str)
            term_match_ratio = term_matches / len(query_terms) if query_terms else 0

            # 词匹配奖励随匹配率增加
            term_match_boost = 1.0 + (0.3 * term_match_ratio)
            scores[item_key]["score"] *= term_match_boost

        # 多检索源奖励
        for item_key, data in scores.items():
            if len(data["matches"]) > 1:
                data["score"] *= 1.25  # 同时出现在多个检索源的项获得25%额外分数

        # 意图相关内容奖励
        if intent_type and intent_type in INTENT_KEYWORDS:
            for item_key, data in scores.items():
                item_str = str(data["item"]).lower()

                # 检查项是否包含相关意图关键词
                relevant_keywords = INTENT_KEYWORDS.get(intent_type, [])
                keyword_matches = sum(1 for kw in relevant_keywords if kw in item_str)

                if keyword_matches > 0:
                    # 最多20%的意图匹配奖励
                    intent_boost = min(1.2, 1.0 + (0.04 * keyword_matches))
                    data["score"] *= intent_boost

        # 按分数排序
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]


# 本地 Sentence Transformer 嵌入函数
class LocalSentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        # 加载本地模型，禁止网络访问
        self.model = SentenceTransformer(
            model_name_or_path=f"/models/sentence-transformers_{model_name}",
            local_files_only=True
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        # 生成嵌入向量
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


# 扩展的Vanna类 - 集成ES和Qdrant向量数据库
class EnhancedVanna(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        self.config = config or {}
        model_name = self.config.get("embedding_model", "bge-m3")

        # 默认文档模式配置
        self.doc_patterns = {
            "table_definition": {
                "pattern": r'# 表名词:(.+?)(?=\n)',
                "block_pattern": r'(# 表名词:.+?)(?=# 表名词:|$)',
                "marker": "# 表名词:"
            },
            "json_data": {
                "pattern": r'## JSON数据（第(\d+)部分，共(\d+)部分）',
                "marker": "## JSON数据"
            },
            "business_scenario": {
                "pattern": r'业务场景:(.*?)(?=\n|$)',
                "marker": "业务场景:"
            },
            "example_data": {
                "pattern": r'示例数据:(.*?)(?=\n|$)',
                "marker": "示例数据:"
            },
            "business_rule": {
                "pattern": r'业务规则:(.*?)(?=\n|$)',
                "marker": "业务规则:"
            }
        }

        # 定义关键标记（用于分块），从配置提取
        self.key_markers = [
            self.doc_patterns["business_scenario"]["marker"],
            self.doc_patterns["example_data"]["marker"],
            self.doc_patterns["business_rule"]["marker"]
        ]

        # 允许通过配置覆盖文档模式
        if "doc_patterns" in self.config:
            for pattern_type, pattern_info in self.config["doc_patterns"].items():
                if pattern_type in self.doc_patterns:
                    self.doc_patterns[pattern_type].update(pattern_info)
                else:
                    self.doc_patterns[pattern_type] = pattern_info

            # 重新生成关键标记列表
            self.key_markers = [
                self.doc_patterns.get("business_scenario", {}).get("marker", "业务场景:"),
                self.doc_patterns.get("example_data", {}).get("marker", "示例数据:"),
                self.doc_patterns.get("business_rule", {}).get("marker", "业务规则:")
            ]

        # 创建本地 SentenceTransformer 嵌入模型实例
        self.embedding_model = SentenceTransformer(
            model_name_or_path=f"/models/sentence-transformers_{model_name}",
            local_files_only=True
        )

        # 配置Qdrant连接参数
        qdrant_config = {
            "url": self.config.get("qdrant_url", "http://localhost:6333"),
            "api_key": self.config.get("qdrant_api_key", None),
            "prefer_grpc": self.config.get("prefer_grpc", True),
            "timeout": self.config.get("qdrant_timeout", 30),
            "sql_collection_name": self.config.get("sql_collection_name", "vanna_sql"),
            "ddl_collection_name": self.config.get("ddl_collection_name", "vanna_ddl"),
            "documentation_collection_name": self.config.get("documentation_collection_name", "vanna_documentation"),
            "n_results": self.config.get("n_results", 10),
        }


        # 初始化Qdrant_VectorStore基类
        Qdrant_VectorStore.__init__(self, config=qdrant_config)
        config.update(qdrant_config)
        self.config = config

        # 初始化OpenAI_Chat基类
        OpenAI_Chat.__init__(self, client=client, config=self.config)

        # 初始化ES客户端（如果配置中指定）
        self.es_client = None
        self.es_indexes = {
            "sql": self.config.get("es_sql_index", "vanna_sql"),
            "ddl": self.config.get("es_ddl_index", "vanna_ddl"),
            "documentation": self.config.get("es_documentation_index", "vanna_documentation")
        }
        self.fusion_weights = {
            "vector": self.config.get("vector_weight", 0.7),
            "bm25": self.config.get("bm25_weight", 0.3)
        }
        self.fusion_method = self.config.get("fusion_method", "rrf")

        # 连接到Elasticsearch
        self._connect_elasticsearch()

        # 猴子补丁原始train方法以确保与ES同步
        self._original_train = self.train
        self.train = self._patched_train

        # 保存检索参数
        self.n_results_sql = self.config.get("n_results_sql", 10)
        self.n_results_ddl = self.config.get("n_results_ddl", 10)
        self.n_results_documentation = self.config.get("n_results_documentation", 10)

    # 重写 generate_embedding 方法，使用我们的本地模型
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        """
        使用本地 SentenceTransformer 模型生成文本嵌入向量

        Args:
            data (str): 需要生成嵌入的文本

        Returns:
            List[float]: 嵌入向量
        """
        try:
            # 使用预先初始化的 embedding_model 生成嵌入向量
            embedding = self.embedding_model.encode(data, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")

    def _connect_elasticsearch(self):
        """连接到Elasticsearch服务器"""
        es_config = self.config.get("elasticsearch", {})
        if not es_config:
            logger.info("未提供Elasticsearch配置，跳过ES初始化")
            return

        try:
            # 创建ES客户端
            self.es_client = Elasticsearch(
                hosts=es_config.get("hosts", ["http://124.71.225.73:9200"]),
                basic_auth=(
                    es_config.get("username", ""),
                    es_config.get("password", "")
                ) if es_config.get("username") else None,
                verify_certs=es_config.get("verify_certs", True),
                timeout=es_config.get("timeout", 30)
            )

            # 验证连接
            if self.es_client.ping():
                logger.info("成功连接到Elasticsearch")

                # 检查并创建索引
                self._create_es_indexes()
            else:
                logger.warning("无法连接到Elasticsearch")
                self.es_client = None
        except Exception as e:
            logger.error(f"连接Elasticsearch时出错: {str(e)}")
            self.es_client = None

    def _create_es_indexes(self):
        """确保所需的ES索引存在"""
        if not self.es_client:
            return

        # 索引映射定义
        mappings = {
            "properties": {
                "document": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "id": {
                    "type": "keyword"
                }
            }
        }

        # 检查并创建索引
        for index_name in self.es_indexes.values():
            if not self.es_client.indices.exists(index=index_name):
                self.es_client.indices.create(
                    index=index_name,
                    body={
                        "mappings": mappings,
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "ik_smart": {
                                        "type": "custom",
                                        "tokenizer": "ik_smart"
                                    },
                                    "ik_max_word": {
                                        "type": "custom",
                                        "tokenizer": "ik_max_word"
                                    }
                                }
                            }
                        }
                    }
                )
                logger.info(f"创建索引 {index_name}")

    def _index_to_es(self, collection_type: str, document: str, id: str):
        """将文档索引到ES"""
        if not self.es_client:
            return

        index_name = self.es_indexes.get(collection_type)
        if not index_name:
            return

        try:
            self.es_client.index(
                index=index_name,
                id=id,
                document={"document": document, "id": id}
            )
            logger.debug(f"文档已索引到 {index_name}, ID: {id}")
        except Exception as e:
            logger.error(f"索引文档到ES时出错: {str(e)}")

    def _search_es(self, collection_type: str, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """从ES中搜索文档"""
        if not self.es_client:
            return []

        index_name = self.es_indexes.get(collection_type)
        if not index_name:
            return []

        try:
            # 使用BM25进行全文搜索
            response = self.es_client.search(
                index=index_name,
                body={
                    "query": {
                        "match": {
                            "document": query
                        }
                    },
                    "size": size
                }
            )

            # 提取结果
            hits = response.get("hits", {}).get("hits", [])
            results = []

            for hit in hits:
                document = hit.get("_source", {}).get("document", "")
                es_score = hit.get("_score", 0)

                # 对于SQL文档，需要解析JSON
                if collection_type == "sql" and document:
                    try:
                        doc_obj = json.loads(document)
                        doc_obj["es_score"] = es_score
                        results.append(doc_obj)
                    except json.JSONDecodeError:
                        results.append({"document": document, "es_score": es_score})
                else:
                    results.append(document)

            return results
        except Exception as e:
            logger.error(f"从ES搜索时出错: {str(e)}")
            return []

    def _fuse_results(self, dense_results: list, bm25_results: list, query: str = "") -> list:
        """使用上下文感知的融合方法融合向量搜索和BM25搜索结果"""
        if not bm25_results:
            return dense_results

        if not dense_results:
            return bm25_results

        method = self.fusion_method.lower()

        if method == "rrf":
            if not query:  # 没有查询上下文时使用基本RRF
                return RankFusion.reciprocal_rank_fusion(
                    [dense_results, bm25_results],
                    weights=[self.fusion_weights["vector"], self.fusion_weights["bm25"]]
                )
            else:  # 有查询上下文时使用上下文感知融合
                return RankFusion.contextual_fusion(
                    query=query,
                    dense_results=dense_results,
                    lexical_results=bm25_results,
                    metadata=None
                )
        elif method == "contextual":
            return RankFusion.contextual_fusion(
                query=query,
                dense_results=dense_results,
                lexical_results=bm25_results,
                metadata=None
            )
        elif method == "multi_stage":
            # 为简化实现，这里使用闭包封装搜索函数
            search_functions = {
                "vector": (lambda q: self._get_vector_results(q, collection_type="sql"), self.fusion_weights["vector"]),
                "bm25": (lambda q: self._search_es("sql", q, size=self.n_results_sql), self.fusion_weights["bm25"])
            }
            return RankFusion.multi_stage_fusion(query, search_functions)
        else:
            # 默认使用基础RRF
            return RankFusion.reciprocal_rank_fusion(
                [dense_results, bm25_results],
                weights=[self.fusion_weights["vector"], self.fusion_weights["bm25"]]
            )

    # 添加新的辅助方法来简化向量检索结果获取
    def _get_vector_results(self, query: str, collection_type: str) -> list:
        """获取向量检索结果的辅助方法"""
        embedding = self.generate_embedding(query)

        if collection_type == "sql":
            results = self._client.query_points(
                self.sql_collection_name,
                query=embedding,
                limit=self.n_results_sql,
                with_payload=True
            ).points

            return [dict(result.payload) for result in results]

        elif collection_type == "ddl":
            results = self._client.query_points(
                self.ddl_collection_name,
                query=embedding,
                limit=self.n_results_ddl,
                with_payload=True
            ).points

            return [result.payload.get("ddl", "") for result in results]

        elif collection_type == "documentation":
            results = self._client.query_points(
                self.documentation_collection_name,
                query=embedding,
                limit=self.n_results_documentation,
                with_payload=True
            ).points

            return [result.payload.get("documentation", "") for result in results]

        return []

    # 重写添加方法以同时添加到ES
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = super().add_question_sql(question, sql, **kwargs)

        # 同时索引到ES
        self._index_to_es("sql", question_sql_json, id)

        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = super().add_ddl(ddl, **kwargs)

        # 同时索引到ES
        self._index_to_es("ddl", ddl, id)

        return id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        增强版添加文档方法，保存块间关联信息
        """
        # 检查是否需要分块
        if len(documentation) > self.config.get("chunk_threshold", 500):
            # 创建分块器实例（保持原有分块器）
            chunker = DocumentChunker(
                mechanical_terms_path=self.config.get("mechanical_terms_path", "/dictionary/MechanicalWords.txt"),
                max_chunk_size=self.config.get("max_chunk_size", 1000),
                overlap=self.config.get("chunk_overlap", 50),
                doc_patterns=self.doc_patterns,  # 传递文档模式配置
                key_markers=self.key_markers  # 传递关键标记
            )

            # 分块
            chunks = chunker.chunk_document(documentation)

            # 分析块之间的关系
            chunks_relations = self._analyze_chunk_relations(chunks, documentation)

            # 对每个分块进行处理
            doc_ids = []
            chunk_id_map = {}  # 保存块文本到ID的映射

            # 第一轮：添加所有块并保存ID
            for i, chunk in enumerate(chunks):
                # 调用父类方法添加文档
                id = super().add_documentation(chunk, **kwargs)

                # 保存ID映射
                chunk_id_map[chunk] = id
                doc_ids.append(id)

                # 同时索引到ES，添加特殊块类型标记
                chunk_type = self._detect_chunk_type(chunk)

                # 构造ES文档，添加元数据
                es_doc = {
                    "document": chunk,
                    "id": id,
                    "chunk_type": chunk_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }

                self._index_to_es("documentation", json.dumps(es_doc), id)

            # 第二轮：更新关联信息
            for chunk, relations in chunks_relations.items():
                if chunk in chunk_id_map:
                    chunk_id = chunk_id_map[chunk]

                    # 将关联ID列表转换为实际ID
                    related_ids = [chunk_id_map[rel_chunk] for rel_chunk in relations
                                   if rel_chunk in chunk_id_map]

                    # 保存关联信息
                    if related_ids:
                        # 更新向量存储中的记录
                        self._update_document_relations(chunk_id, related_ids)

                        # 更新ES中的记录
                        self._update_es_relations("documentation", chunk_id, related_ids)

            # 返回第一个ID
            return doc_ids[0] if doc_ids else ""
        else:
            # 文档较小，不需要分块
            id = super().add_documentation(documentation, **kwargs)

            # 添加元数据
            es_doc = {
                "document": documentation,
                "id": id,
                "chunk_type": "complete_document",
                "chunk_index": 0,
                "total_chunks": 1
            }
            self._index_to_es("documentation", json.dumps(es_doc), id)
            return id

    def _analyze_chunk_relations(self, chunks, full_document):
        """
        分析文档分块间的关联关系

        Args:
            chunks: 文档分块列表
            full_document: 完整文档文本

        Returns:
            dict: 每个块与其相关块的映射关系 {chunk -> [related_chunks]}
        """
        # 初始化结果字典：每个块的关联列表
        relations = {chunk: [] for chunk in chunks}

        # 从配置中获取模式和标记
        table_name_pattern = self.doc_patterns.get("table_definition", {}).get(
            "pattern", r'# 表名词:(.+?)(?=\n)'
        )
        example_data_marker = self.doc_patterns.get("example_data", {}).get(
            "marker", "示例数据:"
        )
        json_pattern = self.doc_patterns.get("json_data", {}).get(
            "pattern", r'## JSON数据（第(\d+)部分，共(\d+)部分）'
        )

        # 获取key_markers用于标记分组
        key_markers = self.key_markers

        # ---------------------------------------
        # 1. 处理表格定义与示例数据的关联
        # ---------------------------------------
        tables = {}  # 表名 -> 块
        examples = {}  # 表名 -> 块

        # 第一轮：识别所有表格定义和示例数据块
        for chunk in chunks:
            # 查找表名
            table_match = re.search(table_name_pattern, chunk)
            if table_match:
                table_name = table_match.group(1).strip()
                tables[table_name] = chunk

            # 查找示例数据引用表名
            for line in chunk.split('\n'):
                if line.startswith(example_data_marker):
                    example_name = line.replace(example_data_marker, "").strip()
                    examples[example_name] = chunk

        # 第二轮：关联表格和示例
        for table_name, table_chunk in tables.items():
            if table_name in examples:
                # 建立双向关联
                relations[table_chunk].append(examples[table_name])
                relations[examples[table_name]].append(table_chunk)
                logger.debug(f"关联表格 '{table_name}' 与其示例数据")

        # ---------------------------------------
        # 2. 处理JSON数据块之间的关联
        # ---------------------------------------
        json_groups = {}  # 组ID -> 块列表

        # 识别所有JSON分块并按组分类
        for chunk in chunks:
            match = re.search(json_pattern, chunk)
            if match:
                part = int(match.group(1))  # 第几部分
                total = int(match.group(2))  # 共几部分
                group_id = f"json_group_{total}"  # 组ID

                if group_id not in json_groups:
                    json_groups[group_id] = []

                json_groups[group_id].append((part, chunk))  # 保存部分编号和块

        # 关联同一组内的所有JSON块
        for group_id, parts_chunks in json_groups.items():
            # 按部分编号排序
            sorted_parts = sorted(parts_chunks, key=lambda x: x[0])

            # 建立组内各块之间的双向关联
            for i, (part1, chunk1) in enumerate(sorted_parts):
                for j, (part2, chunk2) in enumerate(sorted_parts):
                    if i != j:  # 不与自己关联
                        if chunk2 not in relations[chunk1]:
                            relations[chunk1].append(chunk2)
                            logger.debug(f"关联JSON组 {group_id} 的第 {part1} 和第 {part2} 部分")

        # ---------------------------------------
        # 3. 处理同一标记下内容块的关联
        # ---------------------------------------
        # 找出所有标记的位置
        marker_positions = []
        for marker in key_markers:
            for match in re.finditer(re.escape(marker), full_document):
                marker_positions.append((match.start(), marker))

        # 按位置排序
        marker_positions.sort()

        # 为每个块找到其对应的标记
        chunk_markers = {}  # 块 -> 标记

        for chunk in chunks:
            chunk_start = full_document.find(chunk)
            if chunk_start >= 0:
                # 找到最近的前导标记
                nearest_marker = None
                nearest_dist = float('inf')

                for pos, marker in marker_positions:
                    # 标记必须在块之前，且是最近的标记
                    if pos <= chunk_start and chunk_start - pos < nearest_dist:
                        nearest_marker = marker
                        nearest_dist = chunk_start - pos

                # 只记录在合理距离内的标记
                if nearest_marker and nearest_dist < 1000:  # 1000字符的合理距离内
                    chunk_markers[chunk] = nearest_marker
                    logger.debug(f"块与标记 '{nearest_marker}' 关联, 距离: {nearest_dist}")

        # 按标记分组
        marker_groups = {}  # 标记 -> 块列表
        for chunk, marker in chunk_markers.items():
            if marker not in marker_groups:
                marker_groups[marker] = []
            marker_groups[marker].append(chunk)

        # 关联同一标记下的所有块
        for marker, group_chunks in marker_groups.items():
            if len(group_chunks) > 1:  # 只有多个块才需要关联
                for chunk in group_chunks:
                    for other_chunk in group_chunks:
                        # 不与自己关联，且避免重复关联
                        if chunk != other_chunk and other_chunk not in relations[chunk]:
                            relations[chunk].append(other_chunk)
                            logger.debug(f"关联同标记 '{marker}' 下的两个块")

        # ---------------------------------------
        # 4. 特殊关系处理（如业务规则与表的关联)
        # ---------------------------------------
        # 可以在这里添加其他特殊关系的处理逻辑

        return relations


    def _detect_chunk_type(self, chunk):
        """检测块类型，使用配置的模式"""
        # 按照配置尝试各种模式
        for chunk_type, pattern_info in self.doc_patterns.items():
            marker = pattern_info.get("marker", "")
            if marker and marker in chunk:
                return chunk_type

        # 默认检测（如果配置不完整）
        if "# 表名词:" in chunk:
            return "table_definition"
        elif "JSON数据" in chunk:
            return "json_data"
        elif "业务场景:" in chunk:
            return "business_scenario"
        elif "示例数据:" in chunk:
            return "example_data"
        elif "业务规则:" in chunk:
            return "business_rule"
        else:
            return "general"

    def _update_document_relations(self, doc_id, related_ids):
        """更新向量存储中的关联信息"""
        try:
            # 获取现有点信息
            point = self._client.get_points(
                collection_name=self.documentation_collection_name,
                ids=[doc_id]
            ).points[0]

            # 更新payload
            payload = point.payload or {}
            payload["related_chunks"] = related_ids

            # 更新点
            self._client.update_points(
                collection_name=self.documentation_collection_name,
                points=[{"id": doc_id, "payload": payload}]
            )
        except Exception as e:
            logger.error(f"更新文档关联信息失败: {str(e)}")

    def _update_es_relations(self, collection_type, doc_id, related_ids):
        """更新ES中的关联信息"""
        if not self.es_client:
            return

        index_name = self.es_indexes.get(collection_type)
        if not index_name:
            return

        try:
            # 获取现有文档
            doc = self.es_client.get(index=index_name, id=doc_id)
            source = doc.get("_source", {})

            # 更新关联信息
            if isinstance(source, dict):
                source["related_chunks"] = related_ids

                # 更新文档
                self.es_client.update(
                    index=index_name,
                    id=doc_id,
                    doc=source
                )
        except Exception as e:
            logger.error(f"更新ES关联信息失败: {str(e)}")

    # 重写删除方法以同时从ES删除
    def remove_training_data(self, id: str, **kwargs) -> bool:
        result = super().remove_training_data(id, **kwargs)

        # 从ES中删除
        if result and self.es_client:
            collection_type = None
            if id.endswith("-sql"):
                collection_type = "sql"
            elif id.endswith("-ddl"):
                collection_type = "ddl"
            elif id.endswith("-doc"):
                collection_type = "documentation"

            if collection_type:
                index_name = self.es_indexes.get(collection_type)
                if index_name:
                    try:
                        self.es_client.delete(index=index_name, id=id)
                    except Exception as e:
                        logger.error(f"从ES删除文档时出错: {str(e)}")

        return result

    # 重写搜索方法以使用增强的融合算法
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        Enhanced retrieval of similar question-SQL pairs with integrated reranking

        Args:
            question: The user's question
            **kwargs: Additional options including:
              - rerank: Whether to use reranking (default: True)
              - rerank_top_k: Number of candidates to rerank (default: 20)
              - fusion_method: Fusion method to use (default: self.fusion_method)

        Returns:
            List of question-SQL pairs sorted by relevance
        """
        try:
            # Extract options from kwargs with defaults
            use_rerank = kwargs.get('rerank', True)
            rerank_top_k = kwargs.get('rerank_top_k', 20)
            fusion_method = kwargs.get('fusion_method', self.fusion_method)

            # Start with enhanced question preprocessing
            enhanced_question = self.preprocess_field_names(question)

            # 1. Vector search (using original question for semantic relevance)
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "sql")
            vector_time = time.time() - vector_start

            # 2. BM25 search (using enhanced question for better lexical matching)
            bm25_start = time.time()
            bm25_results = self._search_es("sql", enhanced_question, size=self.n_results_sql)
            bm25_time = time.time() - bm25_start

            # Log retrieval statistics
            logger.debug(f"SQL retrieval - Vector: {len(dense_results)} results in {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} results in {bm25_time:.3f}s")

            # Early return if no results from either method
            if not dense_results and not bm25_results:
                logger.warning(f"No SQL results found for question: {question}")
                return []

            # 3. Apply reranking if enabled
            if use_rerank:
                # Combine candidates from both sources for reranking
                seen_ids = set()
                initial_candidates = []

                for result in dense_results + bm25_results:
                    # For SQL documents which are dictionaries with 'question' and 'sql'
                    if isinstance(result, dict):
                        result_id = json.dumps((result.get('question', ''), result.get('sql', '')), sort_keys=True)
                    else:
                        result_id = str(result)

                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        initial_candidates.append(result)

                    if len(initial_candidates) >= rerank_top_k:
                        break

                # 4. Perform direct reranking
                reranked_results = self._rerank_sql_candidates(question, initial_candidates)

                # 5. Apply final contextual fusion if needed
                if fusion_method == "contextual" and reranked_results:
                    final_results = RankFusion.contextual_fusion(
                        query=question,
                        dense_results=reranked_results,  # Use reranked results as dense results
                        lexical_results=bm25_results,  # Keep original lexical results for diversity
                        metadata={"vector_time": vector_time, "bm25_time": bm25_time}
                    )

                    if final_results:
                        return final_results

                # If reranking was successful, return the reranked results
                if reranked_results:
                    return reranked_results

            # 6. Fall back to regular fusion if reranking wasn't used or failed
            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"Error in get_similar_question_sql: {str(e)}")
            logger.error(traceback.format_exc())

            # 修复: 确保变量已定义
            dense_results = locals().get('dense_results', [])
            bm25_results = locals().get('bm25_results', [])

            # Return whatever results we have so far
            return dense_results or bm25_results or []

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        Enhanced retrieval of DDL statements with integrated reranking

        Args:
            question: The user's question
            **kwargs: Additional options including:
              - rerank: Whether to use reranking (default: True)
              - rerank_top_k: Number of candidates to rerank (default: 20)
              - fusion_method: Fusion method to use (default: self.fusion_method)

        Returns:
            List of DDL statements sorted by relevance
        """
        try:
            # Extract options from kwargs with defaults
            use_rerank = kwargs.get('rerank', True)
            rerank_top_k = kwargs.get('rerank_top_k', 20)
            fusion_method = kwargs.get('fusion_method', self.fusion_method)

            # Start with enhanced question preprocessing
            enhanced_question = self.preprocess_field_names(question)

            # Add table name extraction for better DDL matching
            table_names = self._extract_table_names(question)
            if table_names:
                for name in table_names:
                    if name not in enhanced_question:
                        enhanced_question += f" {name}"

            # 1. Vector search (using original question for semantic relevance)
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "ddl")
            vector_time = time.time() - vector_start

            # 2. BM25 search (using enhanced question for better lexical matching)
            bm25_start = time.time()
            bm25_results = self._search_es("ddl", enhanced_question, size=self.n_results_ddl)
            bm25_time = time.time() - bm25_start

            # Log retrieval statistics
            logger.debug(f"DDL retrieval - Vector: {len(dense_results)} results in {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} results in {bm25_time:.3f}s")

            # Early return if no results from either method
            if not dense_results and not bm25_results:
                logger.warning(f"No DDL results found for question: {question}")
                return []

            # 3. Apply reranking if enabled
            if use_rerank:
                # Combine candidates from both sources for reranking
                seen_ddls = set()
                initial_candidates = []

                for ddl in dense_results + bm25_results:
                    ddl_str = str(ddl).strip()
                    if ddl_str and ddl_str not in seen_ddls:
                        seen_ddls.add(ddl_str)
                        initial_candidates.append(ddl_str)

                    if len(initial_candidates) >= rerank_top_k:
                        break

                # 4. Perform direct reranking
                reranked_results = self._rerank_text_documents(question, initial_candidates, "ddl")

                # 5. Apply final contextual fusion if needed
                if fusion_method == "contextual" and reranked_results:
                    final_results = RankFusion.contextual_fusion(
                        query=question,
                        dense_results=reranked_results,  # Use reranked results as dense results
                        lexical_results=bm25_results,  # Keep original lexical results for diversity
                        metadata={"vector_time": vector_time, "bm25_time": bm25_time}
                    )

                    if final_results:
                        return final_results

                # If reranking was successful, return the reranked results
                if reranked_results:
                    return reranked_results

            # 6. Fall back to regular fusion if reranking wasn't used or failed
            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"Error in get_related_ddl: {str(e)}")
            logger.error(traceback.format_exc())

            # 修复: 确保变量已定义
            dense_results = locals().get('dense_results', [])
            bm25_results = locals().get('bm25_results', [])

            # Return whatever results we have so far
            return dense_results or bm25_results or []

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        Enhanced retrieval of documentation with integrated reranking

        Args:
            question: The user's question
            **kwargs: Additional options including:
              - rerank: Whether to use reranking (default: True)
              - rerank_top_k: Number of candidates to rerank (default: 20)
              - fusion_method: Fusion method to use (default: self.fusion_method)

        Returns:
            List of documentation chunks sorted by relevance
        """
        try:
            # Extract options from kwargs with defaults
            use_rerank = kwargs.get('rerank', True)
            rerank_top_k = kwargs.get('rerank_top_k', 20)
            fusion_method = kwargs.get('fusion_method', self.fusion_method)

            # Start with enhanced question preprocessing
            enhanced_question = self.preprocess_field_names(question)

            # Extract business entities and domain terms to enhance docs retrieval
            domain_terms = self._extract_domain_terms(question)
            if domain_terms:
                for term in domain_terms:
                    if term not in enhanced_question:
                        enhanced_question += f" {term}"

            # 1. Vector search (using original question for semantic relevance)
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "documentation")
            vector_time = time.time() - vector_start

            # 2. BM25 search (using enhanced question for better lexical matching)
            bm25_start = time.time()
            bm25_results = self._search_es("documentation", enhanced_question, size=self.n_results_documentation)
            bm25_time = time.time() - bm25_start

            # Log retrieval statistics
            logger.debug(f"Documentation retrieval - Vector: {len(dense_results)} results in {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} results in {bm25_time:.3f}s")

            # Early return if no results from either method
            if not dense_results and not bm25_results:
                logger.warning(f"No documentation results found for question: {question}")
                return []

            # 在返回结果前，添加关联块
            expanded_results = self._expand_with_related_chunks(dense_results + bm25_results)

            # 3. Apply reranking if enabled
            if use_rerank:
                # 使用扩展后的结果创建初始候选集
                seen_docs = set()
                initial_candidates = []

                for doc in expanded_results:  # 使用expanded_results而不是原始结果
                    doc_str = str(doc).strip()
                    if doc_str and doc_str not in seen_docs:
                        seen_docs.add(doc_str)
                        initial_candidates.append(doc_str)

                    if len(initial_candidates) >= rerank_top_k:
                        break

                # 4. Perform direct reranking
                reranked_results = self._rerank_text_documents(question, initial_candidates, "documentation")

                # 5. Apply final contextual fusion if needed
                if fusion_method == "contextual" and reranked_results:
                    final_results = RankFusion.contextual_fusion(
                        query=question,
                        dense_results=reranked_results,  # Use reranked results as dense results
                        lexical_results=expanded_results,  # 使用扩展后结果替代原始bm25_results
                        metadata={"vector_time": vector_time, "bm25_time": bm25_time}
                    )

                    if final_results:
                        return final_results

                # If reranking was successful, return the reranked results
                if reranked_results:
                    return reranked_results

            # 6. Fall back to regular fusion if reranking wasn't used or failed
            return self._fuse_results(dense_results, expanded_results, question)  # 使用expanded_results替代原始bm25_results

        except Exception as e:
            logger.error(f"Error in get_related_documentation: {str(e)}")
            logger.error(traceback.format_exc())
            # 修复: 确保变量已定义
            dense_results = locals().get('dense_results', [])
            bm25_results = locals().get('bm25_results', [])

            # Return whatever results we have so far
            return dense_results or bm25_results or []

    def _expand_with_related_chunks(self, initial_results, max_related=3, max_depth=2, max_total=15):
        """
        扩展检索结果，添加多层关联块

        Args:
            initial_results: 初始检索结果
            max_related: 每个块最多添加的直接相关块数量
            max_depth: 最大递归深度
            max_total: 结果中最大块总数，防止过度扩展

        Returns:
            扩展后的结果列表
        """
        if not initial_results:
            return initial_results

        # 结果去重
        seen_chunks = set()
        expanded = []

        # 跟踪已探索的块和它们的深度
        explored_with_depth = {}

        # 使用广度优先搜索探索关系网络
        queue = [(chunk, 0) for chunk in initial_results]  # (块, 深度)

        while queue and len(expanded) < max_total:
            current_chunk, current_depth = queue.pop(0)

            # 添加当前块（如果尚未添加）
            chunk_str = str(current_chunk)
            if chunk_str not in seen_chunks:
                seen_chunks.add(chunk_str)
                expanded.append(current_chunk)
                explored_with_depth[chunk_str] = current_depth

            # 如果已达到最大深度，不再继续探索
            if current_depth >= max_depth:
                continue

            # 查找相关块
            related_chunks = self._get_related_chunks(current_chunk)
            if not related_chunks:
                continue

            # 限制每个块添加的相关块数量
            related_count = 0
            for related in related_chunks:
                rel_str = str(related)

                # 避免重复探索或添加到队列
                if rel_str in seen_chunks:
                    # 如果已探索但在更深层次，更新为当前更浅层次
                    if rel_str in explored_with_depth and explored_with_depth[rel_str] > current_depth + 1:
                        explored_with_depth[rel_str] = current_depth + 1
                    continue

                # 控制直接相关块数量
                related_count += 1
                if related_count > max_related:
                    break

                # 将相关块加入队列进行后续探索
                queue.append((related, current_depth + 1))

            # 根据相关性和层级对队列排序，优先处理浅层次和高相关性的块
            queue.sort(key=lambda x: x[1])  # 首先按深度排序

        # 记录日志
        logger.debug(f"块关系探索: 初始块数 {len(initial_results)}, 扩展后块数 {len(expanded)}, 最大深度 {max_depth}")

        return expanded

    def _get_related_chunks(self, chunk):
        """获取与给定块相关的块"""
        # 如果是字符串，尝试找到对应的文档ID
        if isinstance(chunk, str):
            chunk_id = self._find_chunk_id(chunk)
            if not chunk_id:
                return []
        else:
            # 如果是字典对象，尝试直接获取ID
            chunk_id = chunk.get("id") if isinstance(chunk, dict) else None
            if not chunk_id:
                return []

        # 从向量存储中获取关联信息
        try:
            point = self._client.get_points(
                collection_name=self.documentation_collection_name,
                ids=[chunk_id]
            ).points[0]

            related_ids = point.payload.get("related_chunks", [])
            if not related_ids:
                return []

            # 获取关联块内容
            related_points = self._client.get_points(
                collection_name=self.documentation_collection_name,
                ids=related_ids
            ).points

            return [point.payload.get("documentation", "") for point in related_points]

        except Exception as e:
            logger.error(f"获取关联块失败: {str(e)}")
            return []

    def _find_chunk_id(self, chunk_text):
        """根据块内容查找ID"""
        # 这里需要实现一个查找机制，可能通过ES或向量存储
        # 简化实现：使用ES进行精确匹配
        if not self.es_client:
            return None

        try:
            response = self.es_client.search(
                index=self.es_indexes.get("documentation"),
                body={
                    "query": {
                        "match_phrase": {
                            "document": chunk_text[:200]  # 使用前200个字符进行匹配
                        }
                    },
                    "size": 1
                }
            )

            hits = response.get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_id")

        except Exception as e:
            logger.error(f"查找块ID失败: {str(e)}")

        return None

    # 添加便利方法，预处理字段名以提高匹配质量
    def preprocess_field_names(self, question: str) -> str:
        """
        预处理问题中的字段名引用，以提高匹配质量
        """
        # 1. 检测可能的字段名（使用引号或驼峰/下划线分隔的词）
        potential_fields = re.findall(r'["\']([^"\']+)["\']|([a-z]+[A-Z][a-zA-Z]*)|([a-z_]+)', question)

        # 展平结果并过滤空项
        potential_fields = [field for group in potential_fields for field in group if field]

        # 2. 为每个潜在字段创建变体
        enhanced_question = question
        for field in potential_fields:
            # 创建常见的字段名变体
            variants = []

            # 驼峰转下划线
            if any(c.isupper() for c in field):
                snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in field]).lstrip('_')
                variants.append(snake_case)

            # 下划线转驼峰
            if '_' in field:
                camel_case = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(field.split('_')))
                variants.append(camel_case)

            # 构造增强后的问题
            for variant in variants:
                if variant not in question:
                    enhanced_question += f" {variant}"

        return enhanced_question

    def _patched_train(self, question: str = None, sql: str = None, ddl: str = None, documentation: str = None,
                       plan=None):
        """
        重写train方法，确保所有添加的训练数据都同步到ES
        """
        logger.info(f"使用增强版train方法，将同步数据到Elasticsearch")

        # 调用原始train方法获取结果
        result = self._original_train(question, sql, ddl, documentation, plan)

        # 确保数据已同步到ES
        if ddl:
            # ddl已经由add_ddl()方法同步，无需额外操作
            pass
        elif documentation:
            # documentation已经由add_documentation()方法同步，无需额外操作
            pass
        elif sql and question:
            # question和sql已经由add_question_sql()方法同步，无需额外操作
            pass
        elif plan:
            # 对于plan，需确保每项内容都同步到ES
            # 注意：这是防御性编程，通常plan.train()已经调用了各个add_*方法
            if self.es_client:
                try:
                    # 执行同步刷新，确保所有文档都已写入ES
                    for index_name in self.es_indexes.values():
                        self.es_client.indices.refresh(index=index_name)
                    logger.info("已刷新所有ES索引")
                except Exception as e:
                    logger.error(f"刷新ES索引时出错: {str(e)}")

        return result

    def get_sql_prompt(
            self,
            initial_prompt: str,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ):
        """增强版SQL提示词生成，提供更好的上下文和指导"""
        if initial_prompt is None:
            initial_prompt = f"你是一位精通{self.dialect}的专家。请基于提供的上下文为以下问题生成SQL查询。\n\n" + \
                             "请特别注意以下几点：\n" + \
                             "1. 适当处理NULL值\n" + \
                             "2. 当涉及多个表时使用正确的JOIN条件\n" + \
                             "3. 使用精确的列名和表引用\n" + \
                             "4. 考虑使用表别名提高可读性\n" + \
                             "5. 应用适当的排序、筛选和聚合函数\n" + \
                             "6. 正确处理字符串比较，考虑大小写敏感性\n" + \
                             "7. 日期时间处理需格式正确\n\n" + \
                             "你的回答应该仅基于给定的上下文，并遵循响应指南。"

        # 增强问题中的字段名称匹配，提高检索质量
        enhanced_question = self.preprocess_field_names(question)

        # 添加DDL和文档上下文
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        # 尝试从DDL提取表关系信息
        table_relationships = self._extract_table_relationships(ddl_list)
        if table_relationships:
            doc_list.append("===表关系信息\n" + table_relationships)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        # 添加更具体的响应指南
        initial_prompt += (
            "===响应指南\n"
            "1. 如果提供的上下文足够，请直接生成有效的SQL查询，无需解释。\n"
            "2. 如果上下文几乎足够但需要特定列中的具体值，请生成一个中间SQL查询来找出该列中的不同值。在查询前添加注释 intermediate_sql\n"
            "3. 如果上下文不足，请解释无法生成的原因。\n"
            "4. 请使用最相关的表。如果需要多个表，确保使用正确的JOIN条件。\n"
            "5. 如果问题之前已被问及并回答，请完全按照之前的方式重复答案。\n"
            f"6. 确保输出的SQL符合{self.dialect}规范，可执行且没有语法错误。\n"
            "7. 当结果需要排序时，使用适当的排序方法。\n"
            "8. 处理文本字段时，考虑大小写敏感性和可能的通配符。\n"
            "9. 处理日期时，确保正确的日期格式和比较方法。\n"
            "10. 对于聚合查询，确保GROUP BY子句包含所有非聚合列。\n"
        )

        message_log = [self.system_message(initial_prompt)]

        # 添加示例
        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        # 添加当前问题
        message_log.append(self.user_message(question))

        return message_log

    def _extract_table_relationships(self, ddl_list):
        """从DDL语句中提取表关系信息，兼容不同数据库"""
        if not ddl_list:
            return ""

        relationships = []

        # 更健壮的正则表达式，可以处理带有CONSTRAINT的定义
        constraint_pattern = r'(?:CONSTRAINT\s+\w+\s+)?FOREIGN KEY\s*\(([^)]+)\)\s*REFERENCES\s+([^(]+)\s*\(([^)]+)\)'

        for ddl in ddl_list:
            fk_matches = re.findall(constraint_pattern, ddl, re.IGNORECASE)

            for match in fk_matches:
                fk_column = match[0].strip()
                ref_table = match[1].strip()
                ref_column = match[2].strip()

                # 清理表名中可能的引号和模式前缀
                ref_table = re.sub(r'["\'\`]', '', ref_table)  # 移除引号
                ref_table = ref_table.split('.')[-1]  # 提取模式限定符后的表名

                # 尝试获取当前表名
                table_match = re.search(r'CREATE TABLE\s+(?:["\'\`])?([^\s"\'`(]+)', ddl, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1).strip()
                    # 清理表名中可能的引号和模式前缀
                    table_name = re.sub(r'["\'\`]', '', table_name)
                    table_name = table_name.split('.')[-1]

                    relationships.append(f"表 {table_name} 的列 {fk_column} 引用 表 {ref_table} 的列 {ref_column}")

        return "\n".join(relationships)

    def _extract_table_names(self, question: str) -> List[str]:
        """
        Extract potential table names from a question

        Args:
            question: The user's question

        Returns:
            List of potential table names
        """
        table_names = []

        # Pattern for common table name references
        patterns = [
            r'(?:table|tables|from|join)\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?',  # "from users" or "join orders"
            r'(?:in|on|for)\s+the\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?\s+table',  # "in the users table"
            r'(?:data|records|rows|information)\s+(?:from|in|of)\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?',  # "data from users"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            table_names.extend([match.lower() for match in matches if match])

        # Include common pluralization/singularization
        expanded_names = set(table_names)
        for name in table_names:
            # Add singular version if plural
            if name.endswith('s'):
                expanded_names.add(name[:-1])
            # Add plural version if singular
            else:
                expanded_names.add(name + 's')

        return list(expanded_names)

    def _extract_domain_terms(self, question: str) -> List[str]:
        """
        Extract business domain terms from a question

        Args:
            question: The user's question

        Returns:
            List of business domain terms
        """
        domain_terms = []

        # Common business entities and their synonyms
        business_entities = {
            "订单": ["order", "purchase", "transaction"],
            "客户": ["customer", "client", "user", "buyer"],
            "产品": ["product", "item", "goods", "merchandise"],
            "支付": ["payment", "transaction", "money", "fund"],
            "价格": ["price", "cost", "amount", "value"],
            "状态": ["status", "state", "condition", "phase"],
            "时间": ["time", "date", "period", "duration"],
            "地址": ["address", "location", "place", "destination"],
            "类型": ["type", "category", "class", "kind"],
            "数量": ["quantity", "amount", "number", "count"]
        }

        # Check for presence of business entities
        for entity, synonyms in business_entities.items():
            if entity in question:
                domain_terms.extend(synonyms)

        # Extract potential technical terms
        tech_patterns = [
            r'(?:table|column|field)\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?',  # "field 'user_id'"
            r'(?:aggregat\w+|calculat\w+|comput\w+)\s+(?:the\s+)?([a-zA-Z0-9_]+)',  # "calculate the total_price"
            r'(?:sort|order|group)\s+by\s+([a-zA-Z0-9_]+)',  # "group by region"
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            domain_terms.extend([match.lower() for match in matches if match])

        return list(set(domain_terms))

    def _rerank_sql_candidates(self, query: str, candidates: List[Any], timeout: int = 10) -> List[Any]:
        """
        Rerank SQL candidates using the reranking service

        Args:
            query: The user's question
            candidates: List of candidate SQL items (dicts with 'question' and 'sql')
            timeout: Timeout for the reranking request in seconds

        Returns:
            Reranked list of candidates or empty list if reranking failed
        """
        if not candidates:
            return []

        # Check if reranking is available
        rerank_url = self.config.get("rerank_url", "http://localhost:8091")

        try:
            # Prepare documents for reranking
            documents = []
            original_items = []

            for candidate in candidates:
                if isinstance(candidate, dict):
                    # Format SQL examples for reranking
                    doc_content = f"Question: {candidate.get('question', '')}\nSQL: {candidate.get('sql', '')}"
                    documents.append({"content": doc_content})
                    original_items.append(candidate)
                else:
                    # Handle non-dict candidates (shouldn't happen for SQL but added for safety)
                    documents.append({"content": str(candidate)})
                    original_items.append(candidate)

            # Call reranking service
            response = requests.post(
                rerank_url,
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": len(documents)  # Rerank all candidates
                },
                timeout=timeout
            )

            if response.status_code == 200:
                # Process reranking results
                reranked_data = response.json()
                reranked_results = []

                # Reconstruct original items in new order with scores
                for item in reranked_data.get("results", []):
                    idx = item.get("index")
                    if 0 <= idx < len(original_items):
                        # If the original item is a dict, add the reranking score
                        if isinstance(original_items[idx], dict):
                            item_copy = original_items[idx].copy()
                            item_copy["rerank_score"] = item.get("score", 0)
                            reranked_results.append(item_copy)
                        else:
                            reranked_results.append(original_items[idx])

                logger.debug(f"Reranked {len(reranked_results)} SQL candidates")
                return reranked_results
            else:
                logger.warning(f"Reranking service returned error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error during SQL reranking: {str(e)}")
            return []

    def _rerank_text_documents(self, query: str, documents: List[str], doc_type: str = "documentation",
                               timeout: int = 10) -> List[str]:
        """
        Rerank text documents using the reranking service

        Args:
            query: The user's question
            documents: List of document strings to rerank
            doc_type: Type of documents ("ddl" or "documentation")
            timeout: Timeout for the reranking request in seconds

        Returns:
            Reranked list of documents or empty list if reranking failed
        """
        if not documents:
            return []

        # Check if reranking is available
        rerank_url = self.config.get("rerank_url", "http://localhost:8091")

        try:
            # Prepare request payload
            # For DDL documents, we might want to extract table names for the reranking query
            enhanced_query = query
            if doc_type == "ddl":
                table_names = self._extract_table_names(query)
                if table_names:
                    enhanced_query = f"{query} table:" + " table:".join(table_names)

            # Call reranking service
            response = requests.post(
                rerank_url,
                json={
                    "query": enhanced_query,
                    "documents": [{"content": doc} for doc in documents],
                    "top_k": len(documents)  # Rerank all documents
                },
                timeout=timeout
            )

            if response.status_code == 200:
                # Process reranking results
                reranked_data = response.json()
                reranked_results = []

                # Reconstruct original documents in new order
                for item in reranked_data.get("results", []):
                    idx = item.get("index")
                    if 0 <= idx < len(documents):
                        reranked_results.append(documents[idx])

                logger.debug(f"Reranked {len(reranked_results)} {doc_type} documents")
                return reranked_results
            else:
                logger.warning(f"Reranking service returned error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error during {doc_type} reranking: {str(e)}")
            return []


from vanna.flask import VannaFlaskApp


# 扩展VannaFlaskApp类来添加自定义API端点
class EnhancedVannaFlaskApp(VannaFlaskApp):
    def __init__(self, vn, *args, **kwargs):
        super().__init__(vn, *args, **kwargs)

        # 存储检索服务实例
        self.retrieval_service = retrieval_service

        # Initialize task manager
        self.task_manager = AsyncTaskManager(max_workers=5)

        # 添加面向dify工作流API端点
        self.add_dify_endpoints()
        # 添加数据管理API端点
        self._add_data_management_endpoints()

    def add_dify_endpoints(self):
        # 添加检索上下文API
        @self.flask_app.route("/api/v0/get_retrieval_context", methods=["GET"])
        @self.requires_auth
        def get_retrieval_context(user: any):
            try:
                # 获取请求参数
                question = request.args.get("question")
                max_results = int(request.args.get("max_results", "10"))
                include_prompt = request.args.get("include_prompt", "true").lower() == "true"
                enhance_query = request.args.get("enhance_query", "true").lower() == "true"
                use_rerank = request.args.get("use_rerank", "true").lower() == "true"
                intent_type = request.args.get("intent_type")  # 新增意图类型参数

                if not question:
                    return jsonify({"type": "error", "error": "No question provided"}), 400

                # 验证intent_type是否有效
                if intent_type and intent_type not in INTENT_KEYWORDS:
                    logger.warning(f"接收到未知意图类型: {intent_type}，将使用自动检测")
                    intent_type = None

                # 使用检索服务执行检索
                options = {
                    "max_results": max_results,
                    "enhance_query": enhance_query,
                    "use_rerank": use_rerank,
                    "intent_type": intent_type  # 传递意图类型
                }

                retrieval_results = self.retrieval_service.retrieve(question, options)

                # 生成提示词（如需要）
                prompt = None
                if include_prompt:
                    prompt_start = time.time()
                    prompt = self.vn.get_sql_prompt(
                        initial_prompt=None,
                        question=question,
                        question_sql_list=retrieval_results["question_sql_list"],
                        ddl_list=retrieval_results["ddl_list"],
                        doc_list=retrieval_results["doc_list"]
                    )
                    retrieval_results["stats"]["prompt_generation_time"] = round(time.time() - prompt_start, 3)

                # 构建响应
                response = {
                    "type": "retrieval_context",
                    "question": question,
                    "context": {
                        "question_sql_list": utils.serialize_data(retrieval_results["question_sql_list"]),
                        "ddl_list": retrieval_results["ddl_list"],
                        "doc_list": retrieval_results["doc_list"],
                        "table_relationships": retrieval_results["table_relationships"]
                    },
                    "dialect": getattr(self.vn, "dialect", "MySQL"),
                    "retrieval_stats": retrieval_results["stats"],
                    "total_time": retrieval_results["total_time"],
                    "detected_intent": retrieval_results.get("detected_intent", None)  # 返回检测到的意图
                }

                if include_prompt:
                    response["prompt"] = prompt

                # 修复后的返回方式 - 使用**将字典展开为关键字参数
                return jsonify(**response)

            except Exception as e:
                logger.error(f"检索API错误: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    "type": "error",
                    "error": str(e),
                    "details": traceback.format_exc() if hasattr(self.vn, "debug") and self.vn.debug else None
                }), 500

        # 添加执行外部SQL的API
        @self.flask_app.route("/api/v0/execute_external_sql", methods=["POST"])
        @self.requires_auth
        def execute_external_sql(user: any):
            try:
                data = request.json
                question = data.get("question")
                sql = data.get("sql")

                if not sql:
                    return jsonify({"type": "error", "error": "No SQL provided"}), 400

                # 生成ID并存储问题和SQL
                id = self.cache.generate_id(question=question)
                self.cache.set(id=id, field="question", value=question)
                self.cache.set(id=id, field="sql", value=sql)

                # 执行SQL
                try:
                    df = self.vn.run_sql(sql=sql)
                    self.cache.set(id=id, field="df", value=df)

                    # 返回结果
                    return jsonify({
                        "type": "df",
                        "id": id,
                        "df": df.head(10).to_json(orient='records', date_format='iso'),
                        "should_generate_chart": self.vn.should_generate_chart(df),
                    })
                except Exception as e:
                    return jsonify({"type": "sql_error", "error": str(e)})

            except Exception as e:
                logger.error(f"执行SQL API错误: {str(e)}")
                return jsonify({"type": "error", "error": str(e)}), 500

    def _add_data_management_endpoints(self):
            """添加数据管理相关的API端点"""

            # 查看提交任务状态
            @self.flask_app.route("/api/v0/task_status", methods=["GET"])
            @self.requires_auth
            def get_task_status(user: any):
                try:
                    task_id = request.args.get("task_id")
                    if not task_id:
                        return jsonify({"type": "error", "error": "No task_id provided"}), 400

                    status = self.task_manager.get_task_status(task_id)
                    return jsonify({
                        "type": "task_status",
                        "task_id": task_id,
                        "status": status
                    })
                except Exception as e:
                    logger.error(f"Task status error: {str(e)}")
                    return jsonify({"type": "error", "error": str(e)}), 500

            # 上传MD文档
            @self.flask_app.route("/api/v0/upload_documentation", methods=["POST"])
            @self.requires_auth
            def upload_documentation(user: any):
                try:
                    # Check if file exists
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # Check file type
                    if not file.filename.endswith(('.md')):
                        return jsonify({"type": "error", "error": "文件类型必须是Markdown"}), 400

                    # Read file content
                    file_content = file.read().decode('utf-8')

                    # Submit document processing as an async task
                    task_id = self.task_manager.submit_task(
                        self._process_document,
                        callback=lambda result: logger.info(
                            f"Document training completed: {len(result) if isinstance(result, list) else 'single'} chunks processed"),
                        file_content=file_content,
                        file_type=file.filename.split('.')[-1]
                    )

                    return jsonify({
                        "type": "task_submitted",
                        "task_id": task_id,
                        "message": "文档上传已开始。系统将在后台处理文档分块和训练。"
                    })
                except Exception as e:
                    logger.error(f"上传文档错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500

            # 上传DDL文件
            @self.flask_app.route("/api/v0/upload_ddl", methods=["POST"])
            @self.requires_auth
            def upload_ddl(user: any):
                try:
                    # Check if file exists
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # Check file type
                    if not file.filename.endswith(('.sql', '.ddl', '.txt')):
                        return jsonify({"type": "error", "error": "文件类型必须是SQL、DDL或TXT"}), 400

                    # Read file content
                    file_content = file.read().decode('utf-8')

                    # Submit DDL processing as an async task
                    task_id = self.task_manager.submit_task(
                        self._process_ddl,
                        callback=lambda result: logger.info(
                            f"DDL training completed: {result['success']} of {result['total']} processed successfully"),
                        file_content=file_content
                    )

                    return jsonify({
                        "type": "task_submitted",
                        "task_id": task_id,
                        "message": "DDL上传已开始。系统将在后台提取和训练表定义。"
                    })
                except Exception as e:
                    logger.error(f"上传DDL错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500

            # 上传SQL示例
            @self.flask_app.route("/api/v0/upload_sql_examples", methods=["POST"])
            @self.requires_auth
            def upload_sql_examples(user: any):
                try:
                    # Check if file exists
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # Check file type
                    if not file.filename.endswith(('.xlsx', '.csv')):
                        return jsonify({"type": "error", "error": "文件类型必须是Excel或CSV"}), 400

                    # Save file temporarily
                    temp_file = os.path.join('/tmp', file.filename)
                    file.save(temp_file)

                    # Submit SQL examples processing as an async task
                    task_id = self.task_manager.submit_task(
                        self._process_sql_examples,
                        callback=lambda result: logger.info(
                            f"SQL examples training completed: {result['success']} of {result['total']} processed successfully"),
                        file_path=temp_file,
                        file_type=file.filename.split('.')[-1]
                    )

                    return jsonify({
                        "type": "task_submitted",
                        "task_id": task_id,
                        "message": "SQL示例上传已开始。系统将在后台处理和训练示例。"
                    })
                except Exception as e:
                    logger.error(f"上传SQL示例错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500
            # 上传术语词典
            @self.flask_app.route("/api/v0/upload_terminology", methods=["POST"])
            @self.requires_auth
            def upload_terminology(user: any):
                try:
                    # 检查是否有文件
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # 检查文件类型
                    if not file.filename.endswith(('.txt', '.xlsx', '.csv')):
                        return jsonify({"type": "error", "error": "文件类型必须是TXT、Excel或CSV"}), 400

                    # 读取并保存文件
                    mechanical_terms_path = os.path.join('/dictionary', 'MechanicalWords.txt')
                    os.makedirs(os.path.dirname(mechanical_terms_path), exist_ok=True)

                    # 处理不同格式
                    if file.filename.endswith('.txt'):
                        file.save(mechanical_terms_path)
                        term_count = sum(1 for _ in open(mechanical_terms_path, 'r', encoding='utf-8').readlines())
                        term_count = term_count // 2  # 一行英文一行中文，所以除以2
                    else:
                        # 对于Excel或CSV，需要转换格式
                        temp_file = os.path.join('/tmp', file.filename)
                        file.save(temp_file)

                        if file.filename.endswith('.xlsx'):
                            df = pd.read_excel(temp_file)
                        else:
                            df = pd.read_csv(temp_file)

                        # 检查必要的列
                        required_columns = ['english', 'chinese']
                        if not all(col in df.columns for col in required_columns):
                            return jsonify({"type": "error", "error": "文件必须包含'english'和'chinese'列"}), 400

                        # 写入到txt文件，一行英文一行中文
                        with open(mechanical_terms_path, 'w', encoding='utf-8') as f:
                            for _, row in df.iterrows():
                                if pd.notna(row['english']) and pd.notna(row['chinese']):
                                    f.write(f"{row['english'].strip()}\n{row['chinese'].strip()}\n")

                        term_count = len(df)
                        os.remove(temp_file)

                    return jsonify({
                        "type": "terminology_upload_result",
                        "path": mechanical_terms_path,
                        "terms": term_count,
                        "status": "success"
                    })
                except Exception as e:
                    logger.error(f"上传术语词典错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500

                # 辅助方法: 按Markdown标题分块

    # 异步处理DDL文档
    def _process_ddl(self, file_content):
        """
        Process DDL content by extracting CREATE TABLE statements
        and training them one by one

        Args:
            file_content: The content of the DDL file

        Returns:
            Dict with processing results
        """
        try:
            # Split into DDL statements
            ddl_statements = re.findall(r'CREATE\s+TABLE\s+.*?;', file_content, re.DOTALL | re.IGNORECASE)

            if not ddl_statements:
                raise ValueError("未找到有效的CREATE TABLE语句")

            # Process each DDL statement
            results = []
            for ddl in ddl_statements:
                try:
                    # Extract table name for logging
                    table_name_match = re.search(r'CREATE\s+TABLE\s+(?:`|")?([^`"\s(]+)', ddl, re.IGNORECASE)
                    table_name = table_name_match.group(1) if table_name_match else "未知表"

                    # Train DDL
                    id = self.vn.train(ddl=ddl)
                    results.append({"table": table_name, "id": id, "status": "success"})
                except Exception as e:
                    results.append({"table": table_name, "error": str(e), "status": "error"})

            return {
                "total": len(ddl_statements),
                "success": sum(1 for r in results if r["status"] == "success"),
                "results": results
            }
        except Exception as e:
            logger.error(f"DDL处理错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

    # 异步处理SQL问题对文档
    def _process_sql_examples(self, file_path, file_type):
        """
        Process SQL examples from Excel or CSV file

        Args:
            file_path: Path to the temporary file
            file_type: File extension (xlsx or csv)

        Returns:
            Dict with processing results
        """
        try:
            # Read file based on type
            if file_type == 'xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Clean up temp file
            try:
                os.remove(file_path)
            except:
                pass

            # Check required columns
            required_columns = ['question', 'sql']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("文件必须包含'question'和'sql'列")

            # Process each row
            results = []
            for i, row in df.iterrows():
                try:
                    if pd.isna(row['question']) or pd.isna(row['sql']):
                        continue

                    question = str(row['question']).strip()
                    sql = str(row['sql']).strip()

                    if not question or not sql:
                        continue

                    # Train question/SQL pair
                    id = self.vn.train(question=question, sql=sql)
                    results.append({
                        "row": i + 1,
                        "question": question[:50] + "..." if len(question) > 50 else question,
                        "id": id,
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "row": i + 1,
                        "question": question[:50] + "..." if len(question) > 50 else question,
                        "error": str(e),
                        "status": "error"
                    })

            return {
                "total": len(df),
                "success": sum(1 for r in results if r["status"] == "success"),
                "results": results
            }
        except Exception as e:
            logger.error(f"SQL示例处理错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

    # 异步处理MD文档
    def _process_document(self, file_content, file_type):
            """
            处理文档内容

            Args:
                file_content: 文件内容
                file_type: 文件类型(md或txt)

            Returns:
                文档ID列表
            """
            try:
                # 直接将整个文档传递给train方法，让它内部处理分块
                doc_id = self.vn.train(documentation=file_content)
                return [doc_id]  # 返回ID列表，方便以后扩展
            except Exception as e:
                logger.error(f"文档处理错误: {str(e)}")
                logger.error(traceback.format_exc())
                raise e
# 主程序
if __name__ == "__main__":
    # 初始化增强版Vanna - 连接到远程ChromaDB和ES
    vn = EnhancedVanna(client=client, config={
        'model': 'qwen-plus',
        'qdrant_url': 'http://124.71.225.73:6333',  # 修改为您的Qdrant服务地址
        'elasticsearch': {
            'hosts': ['http://192.168.222.137:9200']
        },
        'fusion_method': 'contextual',  # 使用上下文感知融合，可选值: rrf, borda, contextual, multi_stage
        'vector_weight': 0.6,  # 向量检索结果权重
        'bm25_weight': 0.4  # BM25检索结果权重
    })

    # 连接到MySQL
    vn.connect_to_mysql(
        host=os.getenv("MYSQL_HOST", "192.168.222.137"),
        dbname=os.getenv("MYSQL_DATABASE", "testdb"),
        user=os.getenv("MYSQL_USER", "testuser"),
        password=os.getenv("MYSQL_PASSWORD", "testpassword"),
        port=int(os.getenv("MYSQL_PORT", "3306"))
    )

    # 初始化检索服务（只创建一次）
    retrieval_service = RetrievalService(vn, {
        "rerank_url": "http://localhost:8091",
        "rerank_enabled": True,
        "max_results": 10
    })

    app = EnhancedVannaFlaskApp(vn)
    app.run(host="0.0.0.0", port=8084)