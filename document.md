# 表格名称:User 

## 业务场景

用于存放用户的基础信息

## 示例数据

| Id(主键,uuid类型) | UserName(用户名称,varchar类型) | Age(年龄,number类型) |
| ----------------- | ------------------------------ | -------------------- |
| xxxx-123-xxx      | 张三                           | 18                   |
| xxxx-456-xxx      | 王五                           | 20                   |

## 业务规则

1. 订单状态由order_type字段标识，有CREATED，PAID，SHIPPED，COMPLETED，CANCELED等类型已取消订单满足:order type='CANCELED'或canceled at不为NULL
2. 已完成订单满足:order_type='COMPLETED'或completed_at不为NULL