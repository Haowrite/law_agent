# 项目描述
基于RAG的法律AI助手，主要基于FastApi与langraph、async构建，基于redis缓存聊天记录。本地部署embedding模型，远程大模型调用阿里百炼的api。


# 环境配置
- 1、ubuntu22.04 环境安装redis，mysql,可以在网上搜索相应的教程。 我的redis版本是8.0，mysql版本是8.0.45。
- 2、anaconda创建环境并执行
```
pip install -r requirements.txt
```
- 3、修改.env_temp文件为.env文件，主要写一个环境配置，具体参数在文件里面有相应描述，初次运行需要设置`RE_BUILD = TRUE`，在自己本地构建`milvus.db`向量库文件。
- 4、根目录添加RAG_DB文件夹用来存储本地向量库，如果没有会报错
- 5、项目根目录命令行执行以下命令运行程序。
```
bash start.sh
```







https://github.com/user-attachments/assets/4bc16df4-f503-45aa-9114-5e1f160a30e5

