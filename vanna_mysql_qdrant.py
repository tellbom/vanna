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

        # for marker in key_markers:
        #     for match in re.finditer(re.escape(marker), protected_text):
        #         marker_positions.append(match.start())

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
    # 在EnhancedVanna类的__init__方法中添加多模型支持
    def __init__(self, client=None, config=None):
        self.config = config or {}

        # ========== 多模型配置 ==========
        # 支持的嵌入模型列表
        self.embedding_models = self.config.get("embedding_models", [
            {
                "name": "bge_m3",
                "service_url": self.config.get("embedding_service_url", "http://localhost:8080"),
                "weight": 0.4,
                "timeout": 30,
                "max_retries": 3
            }
        ])

        # 如果配置了多个模型，使用多模型设置
        if "multi_embedding_services" in self.config:
            self.embedding_models = []
            for model_config in self.config["multi_embedding_services"]:
                self.embedding_models.append({
                    "name": model_config.get("name", "default"),
                    "service_url": model_config.get("service_url"),
                    "weight": model_config.get("weight", 1.0 / len(self.config["multi_embedding_services"])),
                    "timeout": model_config.get("timeout", 30),
                    "max_retries": model_config.get("max_retries", 3)
                })

        # 验证模型配置
        self._validate_embedding_models()

        # ========== Collection名称配置 ==========
        # 为每个模型生成独立的collection名称
        self.model_collections = {}
        base_collections = {
            "sql": self.config.get("sql_collection_name", "vanna_sql"),
            "ddl": self.config.get("ddl_collection_name", "vanna_ddl"),
            "documentation": self.config.get("documentation_collection_name", "vanna_documentation")
        }

        # 为每个模型创建独立的collection映射
        for model in self.embedding_models:
            model_name = model["name"]
            self.model_collections[model_name] = {
                "sql": f"{base_collections['sql']}_{model_name}",
                "ddl": f"{base_collections['ddl']}_{model_name}",
                "documentation": f"{base_collections['documentation']}_{model_name}"
            }

        logger.info(f"配置了 {len(self.embedding_models)} 个嵌入模型")
        logger.info(f"Collection映射: {self.model_collections}")

        # ========== 分数融合配置 ==========
        self.fusion_config = {
            "method": self.config.get("fusion_method", "score_fusion"),  # score_fusion, voting, contextual
            "normalize_scores": self.config.get("normalize_scores", True),
            "diversity_bonus": self.config.get("diversity_bonus", 0.25),  # 多源匹配奖励
            "rrf_k": self.config.get("rrf_k", 30)  # RRF参数
        }

        # ========== 保持原有初始化逻辑 ==========
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

        # ========== Qdrant配置 ==========
        # 使用第一个模型的collection作为默认（向后兼容）
        primary_model = self.embedding_models[0]["name"]
        qdrant_config = {
            "url": self.config.get("qdrant_url", "http://localhost:6333"),
            "api_key": self.config.get("qdrant_api_key", None),
            "prefer_grpc": self.config.get("prefer_grpc", True),
            "timeout": self.config.get("qdrant_timeout", 30),
            "sql_collection_name": self.model_collections[primary_model]["sql"],
            "ddl_collection_name": self.model_collections[primary_model]["ddl"],
            "documentation_collection_name": self.model_collections[primary_model]["documentation"],
            "n_results": self.config.get("n_results", 10),
        }

        # 初始化Qdrant_VectorStore基类
        Qdrant_VectorStore.__init__(self, config=qdrant_config)
        config.update(qdrant_config)
        self.config = config

        # 初始化OpenAI_Chat基类
        OpenAI_Chat.__init__(self, client=client, config=self.config)

        # ========== ES配置保持不变 ==========
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
        self.n_results_sql = self.config.get("n_results_sql", 20)  # 增加检索数量
        self.n_results_ddl = self.config.get("n_results_ddl", 15)
        self.n_results_documentation = self.config.get("n_results_documentation", 25)

        # 在初始化最后添加collection检查和创建
        self._ensure_collections_exist()

        logger.info("多模型EnhancedVanna初始化完成")

    def _ensure_collections_exist(self):
        """确保所有模型的collection都存在"""
        for model_name in self.get_active_models():
            for collection_type in ["sql", "ddl", "documentation"]:
                collection_name = self.get_collection_name(model_name, collection_type)

                try:
                    # 检查collection是否存在
                    if not self._client.collection_exists(collection_name):
                        logger.info(f"创建collection: {collection_name}")

                        # 获取向量维度（需要先生成一个测试向量）
                        test_embedding = self._generate_single_embedding("test", model_name)
                        vector_size = len(test_embedding)

                        # 创建collection
                        from qdrant_client.models import Distance, VectorParams
                        self._client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(f"成功创建collection {collection_name}，向量维度: {vector_size}")
                    else:
                        logger.info(f"Collection {collection_name} 已存在")

                except Exception as e:
                    logger.error(f"处理collection {collection_name} 时出错: {str(e)}")
                    # 从可用模型列表中移除这个模型
                    self.embedding_models = [m for m in self.embedding_models if m["name"] != model_name]

    def _get_vector_results_multi_model(self, query: str, collection_type: str, **kwargs) -> Dict[str, List[Any]]:
        """
        使用多模型进行向量检索

        Args:
            query: 查询文本
            collection_type: collection类型 ("sql", "ddl", "documentation")
            **kwargs: 其他参数

        Returns:
            Dict[str, List[Any]]: 每个模型的检索结果 {model_name: results}
        """
        # 生成多模型嵌入向量
        try:
            query_embeddings = self.generate_embeddings_multi_model(query)
        except Exception as e:
            logger.error(f"生成查询嵌入向量失败: {str(e)}")
            return {}

        # 获取检索数量配置
        limit_map = {
            "sql": self.n_results_sql,
            "ddl": self.n_results_ddl,
            "documentation": self.n_results_documentation
        }
        limit = limit_map.get(collection_type, 10)

        # 搜索参数配置
        search_params = {
            'hnsw_ef': 512,  # 提高搜索精度
            'exact': False,
            'quantization': None,
        }

        all_results = {}

        for model_name, query_embedding in query_embeddings.items():
            try:
                # 获取该模型对应的collection名称
                collection_name = self.get_collection_name(model_name, collection_type)

                # 执行向量搜索
                results = self._client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    with_payload=True,
                    score_threshold=0.0  # 替代with_score
                )
                # 处理结果
                processed_results = []
                for result in results:
                    payload = dict(result.payload)
                    payload["vector_score"] = result.score  # 添加向量相似度分数
                    payload["source_model"] = model_name  # 添加来源模型信息

                    # 根据collection类型提取主要内容
                    if collection_type == "sql":
                        processed_results.append(payload)
                    elif collection_type == "ddl":
                        processed_results.append(payload.get("ddl", ""))
                    elif collection_type == "documentation":
                        processed_results.append(payload.get("documentation", ""))

                all_results[model_name] = processed_results
                logger.debug(f"模型 {model_name} 检索到 {len(processed_results)} 个 {collection_type} 结果")

            except Exception as e:
                logger.error(f"模型 {model_name} 检索失败: {str(e)}")
                all_results[model_name] = []
                continue

        return all_results

    def _validate_embedding_models(self):
        """验证嵌入模型配置"""
        if not self.embedding_models:
            raise ValueError("至少需要配置一个嵌入模型")

        # 验证权重总和
        total_weight = sum(model["weight"] for model in self.embedding_models)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"模型权重总和为 {total_weight}，不等于1.0，将自动标准化")
            # 标准化权重
            for model in self.embedding_models:
                model["weight"] = model["weight"] / total_weight

        # 验证服务连接
        for model in self.embedding_models:
            try:
                response = requests.get(f"{model['service_url']}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ 模型 {model['name']} 服务连接成功: {model['service_url']}")
                else:
                    logger.warning(f"✗ 模型 {model['name']} 服务状态异常: {response.status_code}")
            except Exception as e:
                logger.error(f"✗ 无法连接模型 {model['name']} 服务: {str(e)}")
                raise ConnectionError(f"模型 {model['name']} 服务不可用: {str(e)}")

    def get_active_models(self) -> List[str]:
        """获取当前活跃的模型列表"""
        return [model["name"] for model in self.embedding_models]

    def get_model_weight(self, model_name: str) -> float:
        """获取指定模型的权重"""
        for model in self.embedding_models:
            if model["name"] == model_name:
                return model["weight"]
        return 0.0

    def get_collection_name(self, model_name: str, collection_type: str) -> str:
        """获取指定模型的collection名称"""
        if model_name in self.model_collections:
            return self.model_collections[model_name].get(collection_type, "")
        return ""

    # def _validate_tei_service(self):
    #     """Validate that TEI service is accessible"""
    #     try:
    #         response = requests.get(f"{self.tei_service_url}/health", timeout=5)
    #         if response.status_code == 200:
    #             logger.info(f"TEI service connected successfully at {self.tei_service_url}")
    #         else:
    #             logger.warning(f"TEI service health check failed: {response.status_code}")
    #     except Exception as e:
    #         logger.error(f"Cannot connect to TEI service at {self.tei_service_url}: {str(e)}")
    #         raise ConnectionError(f"TEI service unavailable: {str(e)}")

    # 重写 generate_embedding 方法，使用我们的本地模型
    def generate_embedding(self, data: str, model_name: str = None, **kwargs) -> List[float]:
        """
        使用指定模型生成嵌入向量（单模型版本，保持向后兼容）

        Args:
            data: 要生成嵌入的文本
            model_name: 指定的模型名称，如果为None则使用第一个模型
            **kwargs: 其他参数

        Returns:
            List[float]: 嵌入向量
        """
        if model_name is None:
            model_name = self.embedding_models[0]["name"]

        return self._generate_single_embedding(data, model_name, **kwargs)

    def generate_embeddings_multi_model(self, data: str, **kwargs) -> Dict[str, List[float]]:
        """
        使用所有配置的模型生成嵌入向量

        Args:
            data: 要生成嵌入的文本
            **kwargs: 其他参数

        Returns:
            Dict[str, List[float]]: 每个模型的嵌入向量 {model_name: embedding_vector}
        """
        if not data or not data.strip():
            raise ValueError("输入文本不能为空")

        embeddings = {}
        errors = []

        for model in self.embedding_models:
            model_name = model["name"]
            try:
                embedding = self._generate_single_embedding(data, model_name, **kwargs)
                embeddings[model_name] = embedding
                logger.debug(f"模型 {model_name} 生成嵌入向量成功，维度: {len(embedding)}")
            except Exception as e:
                error_msg = f"模型 {model_name} 生成嵌入向量失败: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if not embeddings:
            raise Exception(f"所有模型都生成嵌入向量失败: {'; '.join(errors)}")

        if len(embeddings) < len(self.embedding_models):
            logger.warning(f"只有 {len(embeddings)}/{len(self.embedding_models)} 个模型成功生成嵌入向量")

        return embeddings

    def _generate_single_embedding(self, data: str, model_name: str, **kwargs) -> List[float]:
        """
        使用指定模型生成单个嵌入向量

        Args:
            data: 要生成嵌入的文本
            model_name: 模型名称
            **kwargs: 其他参数

        Returns:
            List[float]: 嵌入向量
        """
        # 找到对应的模型配置
        model_config = None
        for model in self.embedding_models:
            if model["name"] == model_name:
                model_config = model
                break

        if not model_config:
            raise ValueError(f"未找到模型配置: {model_name}")

        # 准备请求数据
        payload = {
            "inputs": data.strip(),
            "truncate": True
        }

        headers = {
            "Content-Type": "application/json"
        }

        last_exception = None
        max_retries = model_config["max_retries"]
        timeout = model_config["timeout"]
        service_url = model_config["service_url"]

        # 重试逻辑
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = requests.post(
                    f"{service_url}/embed",
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    result = response.json()

                    # 处理不同的返回格式
                    embedding = self._extract_embedding_from_response(result)

                    # 验证嵌入向量
                    if not isinstance(embedding, list) or len(embedding) == 0:
                        raise ValueError(f"无效的嵌入向量格式: {type(embedding)}")

                    elapsed_time = time.time() - start_time
                    logger.debug(
                        f"模型 {model_name} 生成嵌入向量成功，文本长度: {len(data)}, 耗时: {elapsed_time:.3f}s, 维度: {len(embedding)}")

                    return embedding

                elif response.status_code == 413:  # 文本太长
                    if len(data) > 1000:
                        logger.warning(f"模型 {model_name} 文本太长 ({len(data)} 字符)，截断到1000字符")
                        return self._generate_single_embedding(data[:1000], model_name, **kwargs)
                    else:
                        raise Exception(f"模型 {model_name} 文本太长: {response.status_code}")

                elif response.status_code == 422:  # 验证错误
                    error_detail = response.json() if response.content else "未知验证错误"
                    raise ValueError(f"模型 {model_name} 验证错误: {error_detail}")

                else:
                    raise Exception(f"模型 {model_name} 服务错误: {response.status_code} - {response.text}")

            except requests.exceptions.Timeout:
                last_exception = Exception(f"模型 {model_name} 服务超时 {timeout}s (尝试 {attempt + 1})")
                logger.warning(str(last_exception))

            except requests.exceptions.ConnectionError:
                last_exception = Exception(f"模型 {model_name} 连接错误 (尝试 {attempt + 1})")
                logger.warning(str(last_exception))

            except Exception as e:
                last_exception = e
                logger.error(f"模型 {model_name} 嵌入生成错误 (尝试 {attempt + 1}): {str(e)}")

            # 重试前等待（指数退避）
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.info(f"模型 {model_name} 将在 {wait_time}s 后重试...")
                time.sleep(wait_time)

        # 所有重试都失败
        logger.error(f"模型 {model_name} 在 {max_retries} 次尝试后仍然失败")
        raise last_exception or Exception(f"模型 {model_name} 嵌入生成失败")

    def _extract_embedding_from_response(self, result) -> List[float]:
        """
        从TEI响应中提取嵌入向量

        Args:
            result: TEI服务的响应结果

        Returns:
            List[float]: 嵌入向量
        """
        # 处理不同的TEI返回格式
        if isinstance(result, list) and len(result) > 0:
            # 格式1: [embedding_vector]
            return result[0]
        elif isinstance(result, dict) and "embeddings" in result:
            # 格式2: {"embeddings": [embedding_vector]}
            return result["embeddings"][0]
        elif isinstance(result, dict) and "data" in result:
            # 格式3: OpenAI兼容格式 {"data": [{"embedding": [...]}]}
            return result["data"][0]["embedding"]
        else:
            raise ValueError(f"未知的TEI响应格式: {type(result)}")

    def generate_embeddings_batch_multi_model(self, texts: List[str], **kwargs) -> Dict[str, List[List[float]]]:
        """
        批量生成多模型嵌入向量

        Args:
            texts: 文本列表
            **kwargs: 其他参数

        Returns:
            Dict[str, List[List[float]]]: 每个模型的嵌入向量列表 {model_name: [embedding_vectors]}
        """
        if not texts:
            return {}

        results = {}

        for model in self.embedding_models:
            model_name = model["name"]
            try:
                model_embeddings = []
                for text in texts:
                    embedding = self._generate_single_embedding(text, model_name, **kwargs)
                    model_embeddings.append(embedding)
                results[model_name] = model_embeddings
                logger.debug(f"模型 {model_name} 批量生成 {len(texts)} 个嵌入向量成功")
            except Exception as e:
                logger.error(f"模型 {model_name} 批量生成嵌入向量失败: {str(e)}")

        return results

    def test_multi_model_embedding_quality(self, test_pairs: List[tuple] = None):
        """
        测试多模型嵌入向量质量

        Args:
            test_pairs: 测试对列表 [(text1, text2, expected_similarity)]
        """
        import numpy as np

        # 默认测试对
        if test_pairs is None:
            test_pairs = [
                ("SELECT * FROM users", "SELECT * FROM user", 0.8),
                ("CREATE TABLE orders", "DROP TABLE orders", 0.3),
                ("订单状态", "order status", 0.7),
                ("用户表", "users table", 0.7),
            ]

        logger.info("开始测试多模型嵌入向量质量...")

        for text1, text2, expected_sim in test_pairs:
            logger.info(f"\n测试对: '{text1}' vs '{text2}' (期望相似度: {expected_sim})")

            try:
                # 生成多模型嵌入向量
                embeddings1 = self.generate_embeddings_multi_model(text1)
                embeddings2 = self.generate_embeddings_multi_model(text2)

                for model_name in embeddings1:
                    if model_name in embeddings2:
                        emb1 = np.array(embeddings1[model_name])
                        emb2 = np.array(embeddings2[model_name])

                        # 计算余弦相似度
                        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                        # 评估与期望的差异
                        diff = abs(similarity - expected_sim)
                        status = "✓" if diff < 0.2 else "✗"

                        logger.info(f"  {status} 模型 {model_name}: 相似度={similarity:.3f}, 差异={diff:.3f}")

            except Exception as e:
                logger.error(f"测试失败: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str], model_name: str = None, **kwargs) -> List[List[float]]:
        """
        批量生成嵌入向量（单模型版本，保持向后兼容）

        Args:
            texts: 文本列表
            model_name: 指定的模型名称，如果为None则使用第一个模型
            **kwargs: 其他参数

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        if model_name is None:
            model_name = self.embedding_models[0]["name"]

        embeddings = []
        for text in texts:
            try:
                embedding = self._generate_single_embedding(text, model_name, **kwargs)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"批量生成第 {len(embeddings) + 1} 个文本的嵌入向量失败: {str(e)}")
                raise e

        return embeddings

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        获取嵌入向量生成统计信息

        Returns:
            Dict: 统计信息
        """
        stats = {
            "total_models": len(self.embedding_models),
            "active_models": [],
            "model_weights": {},
            "collection_mapping": self.model_collections
        }

        for model in self.embedding_models:
            model_name = model["name"]
            stats["active_models"].append(model_name)
            stats["model_weights"][model_name] = model["weight"]

        return stats

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
                query={"match": {"content": query}},
                size=size,
                track_total_hits=True
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
        """
        获取向量检索结果（保持向后兼容）
        使用多模型检索并融合结果

        Args:
            query: 查询文本
            collection_type: collection类型

        Returns:
            list: 融合后的检索结果
        """
        # 使用多模型检索
        multi_results = self._get_vector_results_multi_model(query, collection_type)

        if not multi_results:
            return []

        # 应用分数融合
        fused_results = self._apply_score_fusion(query, multi_results, collection_type)

        return fused_results

    def _apply_score_fusion(self, query: str, multi_results: Dict[str, List[Any]], collection_type: str) -> List[Any]:
        """
        应用分数融合算法

        Args:
            query: 原始查询
            multi_results: 多模型检索结果
            collection_type: collection类型

        Returns:
            List[Any]: 融合后的结果列表
        """
        if not multi_results:
            return []

        fusion_method = self.fusion_config.get("method", "score_fusion")

        if fusion_method == "score_fusion":
            return self._score_fusion(query, multi_results, collection_type)
        elif fusion_method == "voting":
            return self._voting_fusion(query, multi_results, collection_type)
        elif fusion_method == "contextual":
            return self._contextual_fusion(query, multi_results, collection_type)
        else:
            # 默认使用分数融合
            return self._score_fusion(query, multi_results, collection_type)

    def _score_fusion(self, query: str, multi_results: Dict[str, List[Any]], collection_type: str) -> List[Any]:
        """
        基于分数的融合算法

        Args:
            query: 查询文本
            multi_results: 多模型结果
            collection_type: collection类型

        Returns:
            List[Any]: 融合后的结果
        """
        final_scores = {}

        for model_name, results in multi_results.items():
            model_weight = self.get_model_weight(model_name)

            for rank, item in enumerate(results, start=1):
                # 获取项目的唯一键
                item_key = self._get_fusion_key(item, collection_type)

                # 获取向量相似度分数
                vector_score = 0.0
                if isinstance(item, dict) and "vector_score" in item:
                    vector_score = item["vector_score"]

                # 标准化分数到[0,1]区间
                normalized_score = self._normalize_vector_score(vector_score)

                # 计算加权分数（结合排名和向量分数）
                rank_score = 1.0 / (self.fusion_config.get("rrf_k", 30) + rank)
                combined_score = 0.7 * normalized_score + 0.3 * rank_score
                weighted_score = model_weight * combined_score

                # 累加分数
                if item_key not in final_scores:
                    final_scores[item_key] = {
                        "item": item,
                        "score": 0.0,
                        "sources": set(),
                        "vector_scores": [],
                        "model_ranks": {}
                    }

                final_scores[item_key]["score"] += weighted_score
                final_scores[item_key]["sources"].add(model_name)
                final_scores[item_key]["vector_scores"].append(vector_score)
                final_scores[item_key]["model_ranks"][model_name] = rank

        # 应用多样性奖励
        self._apply_diversity_bonus(final_scores)

        # 排序并返回
        sorted_items = sorted(final_scores.values(), key=lambda x: x["score"], reverse=True)

        # 添加融合元信息
        for item_data in sorted_items:
            if isinstance(item_data["item"], dict):
                item_data["item"]["fusion_score"] = item_data["score"]
                item_data["item"]["source_models"] = list(item_data["sources"])
                item_data["item"]["avg_vector_score"] = sum(item_data["vector_scores"]) / len(
                    item_data["vector_scores"])

        return [item_data["item"] for item_data in sorted_items]

    def _voting_fusion(self, query: str, multi_results: Dict[str, List[Any]], collection_type: str) -> List[Any]:
        """
        基于投票的融合算法

        Args:
            query: 查询文本
            multi_results: 多模型结果
            collection_type: collection类型

        Returns:
            List[Any]: 融合后的结果
        """
        vote_scores = {}

        for model_name, results in multi_results.items():
            model_weight = self.get_model_weight(model_name)

            for rank, item in enumerate(results, start=1):
                item_key = self._get_fusion_key(item, collection_type)

                # 投票分数：排名越前分数越高
                vote_value = model_weight * (len(results) - rank + 1)

                if item_key not in vote_scores:
                    vote_scores[item_key] = {
                        "item": item,
                        "score": 0.0,
                        "votes": {}
                    }

                vote_scores[item_key]["score"] += vote_value
                vote_scores[item_key]["votes"][model_name] = rank

        # 排序并返回
        sorted_items = sorted(vote_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]

    def _contextual_fusion(self, query: str, multi_results: Dict[str, List[Any]], collection_type: str) -> List[Any]:
        """
        上下文感知融合算法
        """
        # 使用原有的RankFusion.contextual_fusion作为基础
        all_results = []
        for results in multi_results.values():
            all_results.extend(results)

        # 简单的上下文融合实现
        return RankFusion.contextual_fusion(
            query=query,
            dense_results=all_results,
            lexical_results=[],  # 这里不使用lexical结果
            k=self.fusion_config.get("rrf_k", 30)
        )

    def _get_fusion_key(self, item: Any, collection_type: str) -> str:
        """
        获取用于融合的唯一键

        Args:
            item: 检索项目
            collection_type: collection类型

        Returns:
            str: 唯一键
        """
        if collection_type == "sql" and isinstance(item, dict):
            # 对于SQL，使用question和sql的组合作为键
            return f"sql_{item.get('question', '')}_{item.get('sql', '')}"
        elif collection_type == "ddl":
            # 对于DDL，使用内容的hash
            return f"ddl_{hash(str(item))}"
        elif collection_type == "documentation":
            # 对于文档，使用内容的hash
            return f"doc_{hash(str(item))}"
        else:
            return f"item_{hash(str(item))}"

    def _normalize_vector_score(self, score: float) -> float:
        """
        标准化向量相似度分数到[0,1]区间

        Args:
            score: 原始分数

        Returns:
            float: 标准化后的分数
        """
        # Qdrant的余弦相似度通常在[-1,1]区间
        # 转换到[0,1]区间
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    def _apply_diversity_bonus(self, final_scores: Dict[str, Dict]):
        """
        应用多样性奖励

        Args:
            final_scores: 最终分数字典
        """
        diversity_bonus = self.fusion_config.get("diversity_bonus", 0.25)
        total_models = len(self.embedding_models)

        for item_key, score_data in final_scores.items():
            # 计算来源多样性
            source_count = len(score_data["sources"])
            diversity_ratio = source_count / total_models

            # 应用多样性奖励
            bonus = diversity_bonus * diversity_ratio
            score_data["score"] *= (1.0 + bonus)

    # 重写添加方法以同时添加到ES
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        增强版添加问题-SQL对，支持多模型存储到独立Collection

        Args:
            question: 用户问题
            sql: 对应的SQL语句
            **kwargs: 其他参数

        Returns:
            str: 主要的文档ID
        """
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )

        # 生成多模型嵌入向量
        try:
            embeddings = self.generate_embeddings_multi_model(question_sql_json)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}")
            raise e

        # 存储到多个独立的Collection
        doc_ids = {}
        primary_id = None

        for model_name, embedding in embeddings.items():
            try:
                # 获取该模型对应的collection名称
                collection_name = self.get_collection_name(model_name, "sql")

                # 构造payload
                payload = {
                    "question": question,
                    "sql": sql,
                    "model_name": model_name,
                    "content_type": "question_sql"
                }

                # 添加到对应的collection
                doc_id = self._add_to_collection(
                    collection_name=collection_name,
                    embedding=embedding,
                    payload=payload,
                    **kwargs
                )

                doc_ids[model_name] = doc_id

                # 使用第一个成功的ID作为主ID
                if primary_id is None:
                    primary_id = doc_id

                logger.debug(f"问题-SQL对已添加到 {collection_name}, ID: {doc_id}")

            except Exception as e:
                logger.error(f"添加到模型 {model_name} 的collection失败: {str(e)}")
                # 继续处理其他模型，不中断整个过程
                continue

        if not doc_ids:
            raise Exception("所有模型的存储都失败了")

        if len(doc_ids) < len(embeddings):
            logger.warning(f"只有 {len(doc_ids)}/{len(embeddings)} 个模型成功存储")

        # 同时索引到ES（保持原有逻辑）
        self._index_to_es("sql", question_sql_json, primary_id)

        logger.info(f"问题-SQL对已存储到 {len(doc_ids)} 个模型的Collection，主ID: {primary_id}")
        return primary_id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        增强版添加DDL，支持多模型存储到独立Collection

        Args:
            ddl: DDL语句
            **kwargs: 其他参数

        Returns:
            str: 主要的文档ID
        """
        # 生成多模型嵌入向量
        try:
            embeddings = self.generate_embeddings_multi_model(ddl)
        except Exception as e:
            logger.error(f"生成DDL嵌入向量失败: {str(e)}")
            raise e

        # 存储到多个独立的Collection
        doc_ids = {}
        primary_id = None

        for model_name, embedding in embeddings.items():
            try:
                # 获取该模型对应的collection名称
                collection_name = self.get_collection_name(model_name, "ddl")

                # 构造payload
                payload = {
                    "ddl": ddl,
                    "model_name": model_name,
                    "content_type": "ddl"
                }

                # 添加到对应的collection
                doc_id = self._add_to_collection(
                    collection_name=collection_name,
                    embedding=embedding,
                    payload=payload,
                    **kwargs
                )

                doc_ids[model_name] = doc_id

                # 使用第一个成功的ID作为主ID
                if primary_id is None:
                    primary_id = doc_id

                logger.debug(f"DDL已添加到 {collection_name}, ID: {doc_id}")

            except Exception as e:
                logger.error(f"添加DDL到模型 {model_name} 的collection失败: {str(e)}")
                continue

        if not doc_ids:
            raise Exception("所有模型的DDL存储都失败了")

        if len(doc_ids) < len(embeddings):
            logger.warning(f"只有 {len(doc_ids)}/{len(embeddings)} 个模型成功存储DDL")

        # 同时索引到ES
        self._index_to_es("ddl", ddl, primary_id)

        logger.info(f"DDL已存储到 {len(doc_ids)} 个模型的Collection，主ID: {primary_id}")
        return primary_id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        增强版添加文档，支持分块和多模型存储

        Args:
            documentation: 文档内容
            **kwargs: 其他参数

        Returns:
            str: 主要的文档ID
        """
        # 检查是否需要分块
        if len(documentation) > self.config.get("chunk_threshold", 500):
            return self._add_documentation_with_chunking(documentation, **kwargs)
        else:
            return self._add_single_documentation(documentation, **kwargs)

    def _add_single_documentation(self, documentation: str, **kwargs) -> str:
        """添加单个文档（不分块）到多模型Collection"""
        # 生成多模型嵌入向量
        try:
            embeddings = self.generate_embeddings_multi_model(documentation)
        except Exception as e:
            logger.error(f"生成文档嵌入向量失败: {str(e)}")
            raise e

        # 存储到多个独立的Collection
        doc_ids = {}
        primary_id = None

        for model_name, embedding in embeddings.items():
            try:
                # 获取该模型对应的collection名称
                collection_name = self.get_collection_name(model_name, "documentation")

                # 构造payload
                payload = {
                    "documentation": documentation,
                    "model_name": model_name,
                    "content_type": "documentation",
                    "chunk_type": "complete_document",
                    "chunk_index": 0,
                    "total_chunks": 1
                }

                # 添加到对应的collection
                doc_id = self._add_to_collection(
                    collection_name=collection_name,
                    embedding=embedding,
                    payload=payload,
                    **kwargs
                )

                doc_ids[model_name] = doc_id

                if primary_id is None:
                    primary_id = doc_id

                logger.debug(f"文档已添加到 {collection_name}, ID: {doc_id}")

            except Exception as e:
                logger.error(f"添加文档到模型 {model_name} 的collection失败: {str(e)}")
                continue

        if not doc_ids:
            raise Exception("所有模型的文档存储都失败了")

        # 同时索引到ES
        es_doc = {
            "document": documentation,
            "id": primary_id,
            "chunk_type": "complete_document",
            "chunk_index": 0,
            "total_chunks": 1
        }
        self._index_to_es("documentation", json.dumps(es_doc), primary_id)

        logger.info(f"文档已存储到 {len(doc_ids)} 个模型的Collection，主ID: {primary_id}")
        return primary_id

    def _add_documentation_with_chunking(self, documentation: str, **kwargs) -> str:
        """添加需要分块的文档到多模型Collection"""
        # 创建分块器实例
        chunker = DocumentChunker(
            mechanical_terms_path=self.config.get("mechanical_terms_path", "/dictionary/MechanicalWords.txt"),
            max_chunk_size=self.config.get("max_chunk_size", 800),  # 使用更小的块大小
            overlap=self.config.get("chunk_overlap", 100),
            doc_patterns=self.doc_patterns,
            key_markers=self.key_markers
        )

        # 分块
        chunks = chunker.chunk_document(documentation)
        logger.info(f"文档已分块为 {len(chunks)} 个部分")

        # 分析块之间的关系
        chunks_relations = self._analyze_chunk_relations(chunks, documentation)

        # 存储所有分块到所有模型
        all_doc_ids = {}  # {model_name: [chunk_ids]}
        chunk_id_maps = {}  # {model_name: {chunk_text: chunk_id}}
        primary_id = None

        # 为每个模型生成所有块的嵌入向量
        for model in self.embedding_models:
            model_name = model["name"]
            try:
                # 批量生成该模型的所有嵌入向量
                embeddings = self.generate_embeddings_batch([chunk for chunk in chunks], model_name)

                # 存储该模型的所有块
                model_chunk_ids = []
                model_chunk_map = {}

                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    # 获取块类型
                    chunk_type = self._detect_chunk_type(chunk)

                    # 构造payload
                    payload = {
                        "documentation": chunk,
                        "model_name": model_name,
                        "content_type": "documentation",
                        "chunk_type": chunk_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }

                    # 获取collection名称
                    collection_name = self.get_collection_name(model_name, "documentation")

                    # 添加到collection
                    doc_id = self._add_to_collection(
                        collection_name=collection_name,
                        embedding=embedding,
                        payload=payload,
                        **kwargs
                    )

                    model_chunk_ids.append(doc_id)
                    model_chunk_map[chunk] = doc_id

                    # 设置主ID
                    if primary_id is None:
                        primary_id = doc_id

                    # 索引到ES
                    es_doc = {
                        "document": chunk,
                        "id": doc_id,
                        "chunk_type": chunk_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "model_name": model_name
                    }
                    self._index_to_es("documentation", json.dumps(es_doc), doc_id)

                all_doc_ids[model_name] = model_chunk_ids
                chunk_id_maps[model_name] = model_chunk_map

                logger.info(f"模型 {model_name} 的 {len(chunks)} 个文档块已存储")

            except Exception as e:
                logger.error(f"模型 {model_name} 的分块存储失败: {str(e)}")
                continue

        if not all_doc_ids:
            raise Exception("所有模型的分块文档存储都失败了")

        # 更新关联信息（对所有成功的模型）
        self._update_multi_model_chunk_relations(chunks_relations, chunk_id_maps)

        logger.info(
            f"分块文档已存储到 {len(all_doc_ids)} 个模型，总计 {sum(len(ids) for ids in all_doc_ids.values())} 个块")
        return primary_id

    def _add_to_collection(self, collection_name: str, embedding: List[float], payload: dict, **kwargs) -> str:
        """
        添加文档到指定的collection

        Args:
            collection_name: collection名称
            embedding: 嵌入向量
            payload: 文档载荷
            **kwargs: 其他参数

        Returns:
            str: 文档ID
        """
        try:
            # 生成唯一ID
            doc_id = self._generate_unique_id()

            # 确保payload是字典格式
            if not isinstance(payload, dict):
                payload = {"content": str(payload)}

            # 构造符合Qdrant格式的点数据
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            )

            # 添加到Qdrant
            self._client.upsert(
                collection_name=collection_name,
                points=[point]
            )

            return doc_id

        except Exception as e:
            logger.error(f"添加到collection {collection_name} 失败: {str(e)}")
            raise e

    def _generate_unique_id(self) -> str:
        """生成唯一的文档ID"""
        import uuid
        return str(uuid.uuid4())

    def _update_multi_model_chunk_relations(self, chunks_relations: dict, chunk_id_maps: dict):
        """
        更新多模型的块关联信息

        Args:
            chunks_relations: 块关系映射 {chunk_text: [related_chunk_texts]}
            chunk_id_maps: 每个模型的块ID映射 {model_name: {chunk_text: chunk_id}}
        """
        for model_name, chunk_id_map in chunk_id_maps.items():
            try:
                collection_name = self.get_collection_name(model_name, "documentation")

                for chunk, relations in chunks_relations.items():
                    if chunk in chunk_id_map:
                        chunk_id = chunk_id_map[chunk]

                        # 获取相关块的ID列表
                        related_ids = [chunk_id_map[rel_chunk]
                                       for rel_chunk in relations
                                       if rel_chunk in chunk_id_map]

                        if related_ids:
                            # 更新该模型collection中的关联信息
                            self._update_collection_relations(collection_name, chunk_id, related_ids)

                logger.debug(f"模型 {model_name} 的块关联信息已更新")

            except Exception as e:
                logger.error(f"更新模型 {model_name} 的块关联信息失败: {str(e)}")

    def _update_collection_relations(self, collection_name: str, doc_id: str, related_ids: List[str]):
        """更新collection中的关联信息"""
        try:
            # 获取现有点信息
            point = self._client.get_points(
                collection_name=collection_name,
                ids=[doc_id]
            ).points[0]

            # 更新payload
            payload = point.payload or {}
            payload["related_chunks"] = related_ids

            # 更新点
            self._client.update_points(
                collection_name=collection_name,
                points=[{"id": doc_id, "payload": payload}]
            )

        except Exception as e:
            logger.error(f"更新collection {collection_name} 中ID {doc_id} 的关联信息失败: {str(e)}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取多模型存储统计信息

        Returns:
            Dict: 存储统计信息
        """
        stats = {
            "models": {},
            "total_collections": 0,
            "collection_names": []
        }

        for model_name in self.get_active_models():
            model_stats = {
                "collections": {},
                "total_documents": 0
            }

            for collection_type in ["sql", "ddl", "documentation"]:
                collection_name = self.get_collection_name(model_name, collection_type)
                try:
                    # 获取collection信息
                    collection_info = self._client.get_collection(collection_name)
                    count = collection_info.points_count

                    model_stats["collections"][collection_type] = {
                        "name": collection_name,
                        "count": count
                    }
                    model_stats["total_documents"] += count

                    stats["collection_names"].append(collection_name)

                except Exception as e:
                    logger.warning(f"获取collection {collection_name} 信息失败: {str(e)}")
                    model_stats["collections"][collection_type] = {
                        "name": collection_name,
                        "count": 0,
                        "error": str(e)
                    }

            stats["models"][model_name] = model_stats
            stats["total_collections"] += len(model_stats["collections"])

        return stats

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
        增强版SQL示例检索，使用多模型融合

        Args:
            question: 用户问题
            **kwargs: 其他参数

        Returns:
            list: 融合后的SQL示例列表
        """
        try:
            # 预处理问题
            enhanced_question = self.preprocess_field_names(question)

            # 多模型向量检索
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "sql")
            vector_time = time.time() - vector_start

            # BM25检索（保持原有逻辑）
            bm25_start = time.time()
            bm25_results = self._search_es("sql", enhanced_question, size=self.n_results_sql)
            bm25_time = time.time() - bm25_start

            logger.debug(f"SQL检索 - 多模型向量: {len(dense_results)} 结果 {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} 结果 {bm25_time:.3f}s")

            # 如果没有结果
            if not dense_results and not bm25_results:
                logger.warning(f"未找到SQL结果: {question}")
                return []

            # 重排序处理
            use_rerank = kwargs.get('rerank', True)
            if use_rerank and dense_results:
                reranked = self._rerank_sql_candidates(question, dense_results[:30])  # 取前30个重排序
                if reranked:
                    dense_results = reranked

            # 融合向量结果和BM25结果
            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"SQL检索失败: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        增强版DDL检索，使用多模型融合
        """
        try:
            enhanced_question = self.preprocess_field_names(question)

            # 添加表名增强
            table_names = self._extract_table_names(question)
            if table_names:
                for name in table_names:
                    if name not in enhanced_question:
                        enhanced_question += f" {name}"

            # 多模型向量检索
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "ddl")
            vector_time = time.time() - vector_start

            # BM25检索
            bm25_start = time.time()
            bm25_results = self._search_es("ddl", enhanced_question, size=self.n_results_ddl)
            bm25_time = time.time() - bm25_start

            logger.debug(f"DDL检索 - 多模型向量: {len(dense_results)} 结果 {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} 结果 {bm25_time:.3f}s")

            if not dense_results and not bm25_results:
                logger.warning(f"未找到DDL结果: {question}")
                return []

            # 重排序处理
            use_rerank = kwargs.get('rerank', True)
            if use_rerank and dense_results:
                reranked = self._rerank_text_documents(question, dense_results[:20], "ddl")
                if reranked:
                    dense_results = reranked

            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"DDL检索失败: {str(e)}")
            return []

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        增强版文档检索，使用多模型融合和关联块扩展
        """
        try:
            enhanced_question = self.preprocess_field_names(question)

            # 添加领域术语增强
            domain_terms = self._extract_domain_terms(question)
            if domain_terms:
                for term in domain_terms:
                    if term not in enhanced_question:
                        enhanced_question += f" {term}"

            # 多模型向量检索
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "documentation")
            vector_time = time.time() - vector_start

            # BM25检索
            bm25_start = time.time()
            bm25_results = self._search_es("documentation", enhanced_question, size=self.n_results_documentation)
            bm25_time = time.time() - bm25_start

            logger.debug(f"文档检索 - 多模型向量: {len(dense_results)} 结果 {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} 结果 {bm25_time:.3f}s")

            if not dense_results and not bm25_results:
                logger.warning(f"未找到文档结果: {question}")
                return []

            # 扩展关联块
            expanded_results = self._expand_with_related_chunks(dense_results + bm25_results)

            # 重排序处理
            use_rerank = kwargs.get('rerank', True)
            if use_rerank and expanded_results:
                reranked = self._rerank_text_documents(question, expanded_results[:25], "documentation")
                if reranked:
                    return reranked

            return self._fuse_results(dense_results, expanded_results, question)

        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return []

    def get_fusion_stats(self) -> Dict[str, Any]:
        """
        获取融合统计信息

        Returns:
            Dict: 融合统计信息
        """
        return {
            "fusion_config": self.fusion_config,
            "active_models": self.get_active_models(),
            "model_weights": {model["name"]: model["weight"] for model in self.embedding_models},
            "collection_mapping": self.model_collections
        }

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
        Rerank SQL candidates using the reranking service (robust version)
        """
        if not candidates:
            return []

        # 1) 读取并规范化 URL
        base = self.config.get("rerank_url", "http://localhost:8091")
        # 如果没有路径，自动补齐到 /rerank
        if base.endswith("/"):
            rerank_url = base + "rerank"
        else:
            # 如果已经显式包含 /rerank 就不重复加
            rerank_url = base if base.rsplit("/", 1)[-1] == "rerank" else base + "/rerank"

        try:
            # 2) 组装文档
            documents = []
            original_items = []
            for cand in candidates:
                if isinstance(cand, dict):
                    doc_content = f"Question: {cand.get('question', '')}\nSQL: {cand.get('sql', '')}"
                    documents.append({"content": doc_content})
                    original_items.append(cand)
                else:
                    documents.append({"content": str(cand)})
                    original_items.append(cand)

            # 3) 可选：预检健康状态（避免 404/连接异常时立刻返回）
            try:
                health_base = base if base.endswith("/health") else (
                    base + "/health" if not base.endswith("/rerank") else base.replace("/rerank", "/health"))
                _ = requests.get(health_base, timeout=3)
            except Exception:
                pass  # 健康检查失败不阻断流程

            # 4) 调用重排服务
            resp = requests.post(
                rerank_url,
                json={"query": query, "documents": documents, "top_k": len(documents)},
                timeout=timeout
            )

            if resp.status_code != 200:
                logger.warning(f"Reranking service returned {resp.status_code} at {rerank_url}")
                return []

            data = resp.json() or {}
            results = data.get("results", [])

            # 5) 兼容两种返回格式
            reranked = []

            # A) 你的老客户端预期：{"results":[{"index":i,"score":s}, ...]}
            if results and isinstance(results[0], dict) and "index" in results[0]:
                for item in results:
                    idx = item.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(original_items):
                        oi = original_items[idx]
                        if isinstance(oi, dict):
                            oi = {**oi, "rerank_score": item.get("score", 0)}
                        reranked.append(oi)
                return reranked

            # B) 你的实际服务返回：排好序的文档/字典，带 rerank_score（见 rerank_service.py）
            #    这里需要把服务返回的每个 doc 映射回原 candidates（通过 content 文本匹配）
            if results and isinstance(results[0], dict):
                # 先构建 content -> index 的映射
                content_to_idx = {}
                for i, doc in enumerate(documents):
                    content_to_idx[doc.get("content", "")] = i

                for doc in results:
                    content = doc.get("content") or doc.get("text") or ""
                    score = doc.get("rerank_score", doc.get("score", 0))
                    idx = content_to_idx.get(content, None)
                    if idx is not None:
                        oi = original_items[idx]
                        if isinstance(oi, dict):
                            oi = {**oi, "rerank_score": score}
                        reranked.append(oi)
                return reranked

            # 兜底：返回原 candidates
            logger.debug("Rerank response shape not recognized; returning original candidates")
            return candidates

        except Exception as e:
            logger.error(f"Error during SQL reranking: {str(e)}")
            return []

    def _rerank_text_documents(
            self,
            query: str,
            documents: List[str],
            doc_type: str = "documentation",
            timeout: int = 10
    ) -> List[str]:
        """
        Rerank text documents using the reranking service (robust version)
        """
        if not documents:
            return []

        # 1) 读取并规范化 URL：自动补全到 /rerank
        base = self.config.get("rerank_url", "http://localhost:8091")
        if base.endswith("/"):
            rerank_url = base + "rerank"
        else:
            rerank_url = base if base.rsplit("/", 1)[-1] == "rerank" else base + "/rerank"

        try:
            # 2) DDL 查询增强（沿用你原有逻辑）
            enhanced_query = query
            if doc_type == "ddl":
                table_names = self._extract_table_names(query)
                if table_names:
                    enhanced_query = f"{query} table:" + " table:".join(table_names)

            # 3) 组织请求体
            req_docs = [{"content": doc} for doc in documents]

            # 4) 可选健康探针（不影响主流程）
            try:
                health_base = base if base.endswith("/health") else (
                    base + "/health" if not base.endswith("/rerank") else base.replace("/rerank", "/health")
                )
                _ = requests.get(health_base, timeout=3)
            except Exception:
                pass

            # 5) 调用重排服务
            resp = requests.post(
                rerank_url,
                json={"query": enhanced_query, "documents": req_docs, "top_k": len(req_docs)},
                timeout=timeout
            )
            if resp.status_code != 200:
                logger.warning(f"Reranking service returned {resp.status_code} at {rerank_url}")
                return []

            data = resp.json() or {}
            results = data.get("results", [])

            # 6) 兼容两种返回格式
            reranked: List[str] = []

            # A) 老格式：{"results":[{"index": i, "score": s}, ...]}
            if results and isinstance(results[0], dict) and "index" in results[0]:
                for item in results:
                    idx = item.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(documents):
                        reranked.append(documents[idx])
                logger.debug(f"Reranked {len(reranked)} {doc_type} documents (by index)")
                return reranked

            # B) 实际服务：返回排好序的文档字典（含 rerank_score/score）
            if results and isinstance(results[0], dict):
                # 建 content -> 原始索引 映射（用我们发送时的 content 串）
                content_to_idx = {d.get("content", ""): i for i, d in enumerate(req_docs)}
                for doc in results:
                    content = doc.get("content") or doc.get("text") or ""
                    idx = content_to_idx.get(content, None)
                    if idx is not None:
                        reranked.append(documents[idx])
                logger.debug(f"Reranked {len(reranked)} {doc_type} documents (by content)")
                return reranked

            # 兜底：返回原序
            logger.debug("Rerank response shape not recognized; returning original order")
            return documents

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
                # 或者使用更安全的方法
                question = request.args.get("question", "")
                if question:
                    # 尝试修复编码问题
                    try:
                        question = question.encode('iso-8859-1').decode('utf-8')
                    except:
                        print("用户问题："+ question)
                        pass  # 如果解码失败，使用原始字符串

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
def create_multi_model_config():
    """
    创建多模型配置的示例函数

    Returns:
        Dict: 完整的多模型配置
    """

    # 基础配置
    base_config = {
        'model': 'qwen-plus',
        'qdrant_url': 'http://192.168.48.128:6333',

        # ========== 多模型嵌入服务配置 ==========
        'multi_embedding_services': [
            {
                'name': 'bge_m3',
                'service_url': 'http://192.168.48.128:6206',
                'weight': 0.4,
                'timeout': 30,
                'max_retries': 3,
                'description': 'BGE-M3多语言模型，擅长代码和通用文本'
            },
            {
                'name': 'text2vec_chinese',
                'service_url': 'http://192.168.48.128:6006',
                'weight': 0.35,
                'timeout': 30,
                'max_retries': 3,
                'description': '中文文本向量模型，擅长中文业务文档'
            },
            {
                'name': 'sentence_transformers',
                'service_url': 'http://192.168.48.128:6106',
                'weight': 0.25,
                'timeout': 30,
                'max_retries': 3,
                'description': 'Sentence-BERT模型，提供多样性补充'
            }
        ],

        # ========== Collection名称配置 ==========
        'sql_collection_name': 'vanna_sql',
        'ddl_collection_name': 'vanna_ddl',
        'documentation_collection_name': 'vanna_documentation',

        # ========== 分数融合配置 ==========
        'fusion_method': 'score_fusion',  # score_fusion, voting, contextual
        'normalize_scores': True,
        'diversity_bonus': 0.25,  # 多源匹配奖励25%
        'rrf_k': 30,  # RRF参数

        # ========== 检索精度优化配置 ==========
        'n_results_sql': 30,  # SQL示例检索数量：增加到30
        'n_results_ddl': 20,  # DDL检索数量：增加到20
        'n_results_documentation': 35,  # 文档检索数量：增加到35

        # ========== 文档分块优化配置 ==========
        'chunk_threshold': 300,  # 更小的分块阈值：500 -> 300
        'max_chunk_size': 800,  # 更小的块大小：1000 -> 800
        'chunk_overlap': 100,  # 保持较大重叠

        # ========== Elasticsearch配置 ==========
        'elasticsearch': {
            'hosts': ['http://192.168.48.128:9200']
        },

        # ========== 融合权重配置 ==========
        'vector_weight': 0.75,  # 提高向量检索权重：0.6 -> 0.75
        'bm25_weight': 0.25,  # 降低BM25权重：0.4 -> 0.25

        # ========== 重排序配置 ==========
        'rerank_url': 'http://192.168.48.128:8091',
        'rerank_enabled': True,
        'rerank_timeout': 60,

        # ========== 质量控制配置 ==========
        'enable_quality_check': True,
        'quality_threshold': 0.6,  # 检索质量阈值
        'max_fusion_candidates': 50,  # 融合算法最大候选数量

        # ========== 性能配置 ==========
        'enable_caching': True,
        'cache_ttl': 3600,  # 缓存1小时
        'batch_size': 32,

        # ========== 专业术语配置 ==========
        'mechanical_terms_path': '/dictionary/MechanicalWords.txt',
    }

    return base_config


def create_single_model_config():
    """
    创建单模型配置（向后兼容）

    Returns:
        Dict: 单模型配置
    """
    return {
        'model': 'qwen-plus',
        'qdrant_url': 'http://192.168.48.128:6333',

        # 单模型配置
        'embedding_service_url': 'http://192.168.48.128:8080',
        'embedding_timeout': 30,
        'embedding_max_retries': 3,

        'elasticsearch': {
            'hosts': ['http://192.168.48.128:9200']
        },
        'fusion_method': 'contextual',
        'vector_weight': 0.7,
        'bm25_weight': 0.3,
        'rerank_url': 'http://192.168.48.128:8091',
        'rerank_enabled': True,
    }


def setup_logging():
    """配置日志系统"""
    import logging
    import os

    # 确保日志目录存在
    log_dir = '/logs'
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError:
            # 如果无法创建/logs，使用当前目录
            log_dir = '/app/logs'
            os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'vanna_multi_model.log')

    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(log_file, encoding='utf-8')  # 文件输出
        ]
    )

    # 设置特定模块的日志级别
    logging.getLogger("EnhancedVanna").setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def validate_multi_model_setup(config):
    """
    验证多模型配置的有效性

    Args:
        config: 配置字典

    Returns:
        bool: 验证是否通过
    """
    logger = logging.getLogger("ConfigValidator")

    # 检查必要的配置项
    required_keys = ['qdrant_url', 'multi_embedding_services']
    for key in required_keys:
        if key not in config:
            logger.error(f"缺少必要配置项: {key}")
            return False

    # 检查嵌入服务配置
    services = config.get('multi_embedding_services', [])
    if not services:
        logger.error("至少需要配置一个嵌入服务")
        return False

    total_weight = sum(service.get('weight', 0) for service in services)
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"模型权重总和为 {total_weight}，将自动标准化到1.0")

    # 检查服务连接
    for service in services:
        name = service.get('name', 'unknown')
        url = service.get('service_url')

        if not url:
            logger.error(f"服务 {name} 缺少service_url配置")
            return False

        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ 服务 {name} 连接正常: {url}")
            else:
                logger.warning(f"✗ 服务 {name} 状态异常: {response.status_code}")
        except Exception as e:
            logger.error(f"✗ 无法连接服务 {name}: {str(e)}")
            return False

    logger.info("多模型配置验证通过")
    return True


def initialize_enhanced_vanna(use_multi_model=True):
    """
    初始化增强版Vanna实例

    Args:
        use_multi_model: 是否使用多模型模式

    Returns:
        EnhancedVanna: 初始化后的实例
    """
    # 设置日志
    setup_logging()
    logger = logging.getLogger("VannaInitializer")

    # 选择配置
    if use_multi_model:
        config = create_multi_model_config()
        logger.info("使用多模型配置")

        # 验证配置
        if not validate_multi_model_setup(config):
            logger.error("多模型配置验证失败，退出")
            exit(1)
    else:
        config = create_single_model_config()
        logger.info("使用单模型配置（向后兼容模式）")

    # 初始化Vanna实例
    try:
        vn = EnhancedVanna(client=client, config=config)
        logger.info("EnhancedVanna初始化成功")

        # 连接MySQL
        vn.connect_to_mysql(
            host=os.getenv("MYSQL_HOST", "192.168.48.128"),
            dbname=os.getenv("MYSQL_DATABASE", "testdb"),
            user=os.getenv("MYSQL_USER", "testuser"),
            password=os.getenv("MYSQL_PASSWORD", "testpassword"),
            port=int(os.getenv("MYSQL_PORT", "3306"))
        )
        logger.info("MySQL连接成功")

        # 测试嵌入向量生成
        if use_multi_model:
            vn.test_multi_model_embedding_quality()

            # 打印统计信息
            stats = vn.get_embedding_stats()
            logger.info(f"活跃模型: {stats['active_models']}")
            logger.info(f"模型权重: {stats['model_weights']}")

        return vn

    except Exception as e:
        logger.error(f"EnhancedVanna初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        exit(1)


def create_retrieval_service(vn):
    """
    创建检索服务实例

    Args:
        vn: Vanna实例

    Returns:
        RetrievalService: 检索服务实例
    """
    logger = logging.getLogger("RetrievalServiceCreator")

    try:
        retrieval_service = RetrievalService(vn, {
            "rerank_url": "http://192.168.48.128:8091",
            "rerank_enabled": True,
            "max_results": 25,  # 增加检索结果数量
            "rerank_timeout": 60
        })

        logger.info("检索服务初始化成功")
        return retrieval_service

    except Exception as e:
        logger.error(f"检索服务初始化失败: {str(e)}")
        raise e


def print_startup_info(vn, retrieval_service):
    """
    打印启动信息

    Args:
        vn: Vanna实例
        retrieval_service: 检索服务实例
    """
    logger = logging.getLogger("StartupInfo")

    print("\n" + "=" * 80)
    print("🚀 多模型增强版Vanna启动成功！")
    print("=" * 80)

    # 模型信息
    if hasattr(vn, 'embedding_models') and len(vn.embedding_models) > 1:
        print(f"📊 多模型模式: {len(vn.embedding_models)} 个嵌入模型")
        for model in vn.embedding_models:
            print(f"   - {model['name']}: 权重 {model['weight']:.2f}, 服务 {model['service_url']}")
    else:
        print("📊 单模型模式（向后兼容）")

    # Collection信息
    if hasattr(vn, 'model_collections'):
        total_collections = sum(len(collections) for collections in vn.model_collections.values())
        print(f"💾 独立Collection: {total_collections} 个")
        for model_name, collections in vn.model_collections.items():
            print(f"   - 模型 {model_name}: {list(collections.values())}")

    # 融合配置
    if hasattr(vn, 'fusion_config'):
        print(f"🔀 融合算法: {vn.fusion_config.get('method', 'default')}")
        print(f"🎯 多样性奖励: {vn.fusion_config.get('diversity_bonus', 0) * 100:.0f}%")

    # API端点
    print("🌐 API端点:")
    print("   - 检索上下文: GET /api/v0/get_retrieval_context")
    print("   - 执行SQL: POST /api/v0/execute_external_sql")
    print("   - 上传文档: POST /api/v0/upload_documentation")
    print("   - 上传DDL: POST /api/v0/upload_ddl")
    print("   - 任务状态: GET /api/v0/task_status")

    print("=" * 80)
    print("📖 使用指南:")
    print("   1. 访问 http://localhost:8084 查看Web界面")
    print("   2. 使用API端点进行程序化访问")
    print("   3. 查看日志文件: /logs/vanna_multi_model.log")
    print("=" * 80 + "\n")


# 主程序入口
if __name__ == "__main__":
    # 环境变量配置
    USE_MULTI_MODEL = os.getenv("USE_MULTI_MODEL", "true").lower() == "true"
    PORT = int(os.getenv("VANNA_PORT", "8084"))
    HOST = os.getenv("VANNA_HOST", "0.0.0.0")

    try:
        # 1. 初始化增强版Vanna
        vn = initialize_enhanced_vanna(use_multi_model=USE_MULTI_MODEL)

        # 2. 创建检索服务
        retrieval_service = create_retrieval_service(vn)

        # 3. 创建Flask应用
        app = EnhancedVannaFlaskApp(vn)
        app.config['JSON_AS_ASCII'] = False  # 允许非ASCII字符

        # 4. 打印启动信息
        print_startup_info(vn, retrieval_service)

        # 5. 启动服务
        logger = logging.getLogger("Main")
        logger.info(f"启动Web服务: http://{HOST}:{PORT}")
        app.run(host=HOST, port=PORT, debug=False)

    except KeyboardInterrupt:
        print("\n👋 用户中断，正在关闭服务...")
        exit(0)
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        logging.getLogger("Main").error(traceback.format_exc())
        exit(1)