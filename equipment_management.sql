-- 设备管理系统数据库结构
-- 创建部门信息表（先创建被引用表）
CREATE TABLE department_info (
    department_id VARCHAR(20) PRIMARY KEY,
    department_name VARCHAR(50) NOT NULL,
    manager VARCHAR(50),
    contact_phone VARCHAR(20),
    location VARCHAR(100)
);

-- 创建设备信息表
CREATE TABLE equipment_info (
    equipment_id VARCHAR(20) PRIMARY KEY,
    equipment_name VARCHAR(100) NOT NULL,
    model VARCHAR(50),
    manufacturer VARCHAR(100),
    purchase_date DATE,
    installation_date DATE,
    service_life INT COMMENT '预计使用年限(年)',
    status ENUM('正常', '维修中', '停用'),
    department_id VARCHAR(20),
    location VARCHAR(100),
    last_maintenance_date DATE,
    FOREIGN KEY (department_id) REFERENCES department_info(department_id)
);

-- 创建维护记录表
CREATE TABLE maintenance_record (
    maintenance_id VARCHAR(20) PRIMARY KEY,
    equipment_id VARCHAR(20),
    maintenance_type ENUM('定期维护', '故障维修'),
    start_date DATETIME,
    end_date DATETIME,
    maintenance_staff VARCHAR(50),
    fault_description TEXT,
    maintenance_content TEXT,
    parts_replaced TEXT COMMENT '多个零部件用逗号分隔',
    maintenance_cost DECIMAL(10,2) COMMENT '单位：元',
    status ENUM('计划中', '进行中', '已完成'),
    FOREIGN KEY (equipment_id) REFERENCES equipment_info(equipment_id)
);