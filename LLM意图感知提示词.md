# 意图识别步骤
system: 请判断以下用户问题属于哪种查询意图。仅回复单个意图代码，不要有其他内容。
意图代码列表：
- general: 通用查询
- status: 状态查询
- time: 时间查询
- type: 类型查询
- count: 数量查询
- aggregation: 聚合查询
- comparison: 比较查询
- equipment: 设备查询
- component: 零部件查询
- material: 材料查询
- specification: 规格参数查询
- process: 工艺流程查询
- maintenance: 维护保养查询
- troubleshooting: 故障诊断查询
- quality: 质量控制查询
- efficiency: 效率查询
- cost: 成本查询
- safety: 安全规范查询
- supply_chain: 供应链查询

用户问题: "{{$user_input}}"