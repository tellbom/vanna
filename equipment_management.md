# 生产设备管理系统

业务场景:设备维护管理
本场景主要涉及工厂设备的基本信息管理、定期维护计划安排以及故障维修记录追踪。通过系统的数据分析，可以预测设备可能出现的故障，提前安排维护，延长设备使用寿命，降低维修成本。

# 表名词:设备信息表
设备信息表存储了所有生产设备的基本信息，是整个系统的核心表。

| 字段名 | 数据类型 | 说明 | 备注 |
|--------|----------|------|------|
| equipment_id | VARCHAR(20) | 设备编号 | 主键 |
| equipment_name | VARCHAR(100) | 设备名称 | 非空 |
| model | VARCHAR(50) | 型号 | |
| manufacturer | VARCHAR(100) | 制造商 | |
| purchase_date | DATE | 采购日期 | |
| installation_date | DATE | 安装日期 | |
| service_life | INT | 预计使用年限 | 单位：年 |
| status | ENUM | 设备状态 | 正常、维修中、停用 |
| department_id | VARCHAR(20) | 所属部门 | 外键 |
| location | VARCHAR(100) | 设备位置 | |
| last_maintenance_date | DATE | 上次维护日期 | |

# 表名词:维护记录表
维护记录表记录了设备的所有维护活动，包括定期维护和故障维修。

| 字段名 | 数据类型 | 说明 | 备注 |
|--------|----------|------|------|
| maintenance_id | VARCHAR(20) | 维护记录ID | 主键 |
| equipment_id | VARCHAR(20) | 设备编号 | 外键，关联设备信息表 |
| maintenance_type | ENUM | 维护类型 | 定期维护、故障维修 |
| start_date | DATETIME | 开始时间 | |
| end_date | DATETIME | 结束时间 | |
| maintenance_staff | VARCHAR(50) | 维护人员 | |
| fault_description | TEXT | 故障描述 | 故障维修时填写 |
| maintenance_content | TEXT | 维护内容 | |
| parts_replaced | TEXT | 更换零部件 | 多个零部件用逗号分隔 |
| maintenance_cost | DECIMAL(10,2) | 维护费用 | 单位：元 |
| status | ENUM | 维护状态 | 计划中、进行中、已完成 |

# 表名词:部门信息表
部门信息表存储了公司各个部门的基本信息。

| 字段名 | 数据类型 | 说明 | 备注 |
|--------|----------|------|------|
| department_id | VARCHAR(20) | 部门编号 | 主键 |
| department_name | VARCHAR(50) | 部门名称 | 非空 |
| manager | VARCHAR(50) | 部门主管 | |
| contact_phone | VARCHAR(20) | 联系电话 | |
| location | VARCHAR(100) | 部门位置 | |

示例数据:设备信息表
```json
[
  {
    "equipment_id": "EQ001",
    "equipment_name": "数控车床",
    "model": "CK6136",
    "manufacturer": "沈阳机床厂",
    "purchase_date": "2020-03-15",
    "installation_date": "2020-03-20",
    "service_life": 10,
    "status": "正常",
    "department_id": "D001",
    "location": "一号车间A区",
    "last_maintenance_date": "2023-07-10"
  },
  {
    "equipment_id": "EQ002",
    "equipment_name": "立式加工中心",
    "model": "VMC850",
    "manufacturer": "大连机床集团",
    "purchase_date": "2019-05-20",
    "installation_date": "2019-06-01",
    "service_life": 12,
    "status": "维修中",
    "department_id": "D001",
    "location": "一号车间B区",
    "last_maintenance_date": "2023-06-15"
  },
  {
    "equipment_id": "EQ003",
    "equipment_name": "数控铣床",
    "model": "XK7136",
    "manufacturer": "沈阳机床厂",
    "purchase_date": "2021-02-10",
    "installation_date": "2021-02-15",
    "service_life": 10,
    "status": "正常",
    "department_id": "D002",
    "location": "二号车间A区",
    "last_maintenance_date": "2023-08-05"
  }
]
```

示例数据:设备信息表
```json
[
  {
    "maintenance_id": "M001",
    "equipment_id": "EQ001",
    "maintenance_type": "定期维护",
    "start_date": "2023-07-10 08:00:00",
    "end_date": "2023-07-10 12:00:00",
    "maintenance_staff": "张工",
    "fault_description": "",
    "maintenance_content": "按计划进行日常维护，检查各部件运转情况，更换润滑油",
    "parts_replaced": "润滑油",
    "maintenance_cost": 500.00,
    "status": "已完成"
  },
  {
    "maintenance_id": "M002",
    "equipment_id": "EQ002",
    "maintenance_type": "故障维修",
    "start_date": "2023-06-15 13:30:00",
    "end_date": null,
    "maintenance_staff": "李工",
    "fault_description": "主轴箱异响，切削精度下降",
    "maintenance_content": "更换主轴轴承，调整主轴预紧力",
    "parts_replaced": "主轴轴承,油封",
    "maintenance_cost": 3500.00,
    "status": "进行中"
  },
  {
    "maintenance_id": "M003",
    "equipment_id": "EQ003",
    "maintenance_type": "定期维护",
    "start_date": "2023-08-05 09:00:00",
    "end_date": "2023-08-05 11:30:00",
    "maintenance_staff": "王工",
    "fault_description": "",
    "maintenance_content": "检查电气系统，清理冷却系统",
    "parts_replaced": "冷却液滤芯",
    "maintenance_cost": 300.00,
    "status": "已完成"
  }
]
```
示例数据:部门信息表
```json
[
  {
    "department_id": "D001",
    "department_name": "机加工部",
    "manager": "刘主管",
    "contact_phone": "13812345678",
    "location": "一号厂房"
  },
  {
    "department_id": "D002",
    "department_name": "装配部",
    "manager": "赵主管",
    "contact_phone": "13987654321",
    "location": "二号厂房"
  },
  {
    "department_id": "D003",
    "department_name": "质检部",
    "manager": "钱主管",
    "contact_phone": "13600112233",
    "location": "综合楼一层"
  }
]
```

# 业务规则:设备维护管理规则

每台设备必须至少每季度进行一次定期维护
设备状态为"维修中"时，不能安排新的维护任务
对于服务年限超过预计使用年限的设备，维护频率应提高到每月一次
维护费用超过设备原值30%的，应评估是否需要更换设备
同一设备连续三次因相同部件故障进行维修的，应进行根本原因分析
设备所属部门变更时，需同步更新设备位置信息
维护记录必须包含详细的维护内容和更换的零部件信息
故障维修完成后，必须更新设备信息表中的最后维护日期
