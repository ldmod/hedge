# 使用方式
```
demo: cryptoqt.data.data_manager.py

import cryptoqt.data.data_manager as dm
dm.init()
dr=dm.dr
```
# 时间格式
## 1、格式化int存储
```
tools.tmu2i(tm) unix时间戳转int
tools.tmi2u(tm) int转unix时间戳
```
## 2、unix时间戳
## 3、分钟索引，距离g_start_tm=20210101080000多少分钟
```
get_minidx， 根据unix时间戳获取分钟索引
get_tm_by_minidx， 根据分钟索引获取unix时间戳
lasted_minidx， 获取分钟索引最大值（最新值）
```
## 4、秒级索引，距离g_start_tm=20240101080000多少秒
```
get_secidx， 根据unix时间戳获取秒级索引
get_tm_by_secidx， 根据秒级索引获取unix时间戳
lasted_secidx， 获取秒级索引最大值（最新值）
```
