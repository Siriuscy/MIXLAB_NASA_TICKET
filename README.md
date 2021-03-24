# MIXLAB_NASA_TICKET
mixlab

灵感来源于NASA的火星船票，我们想要使用开源的代码来定制化这一设计。   
其中photo_to_cartoon 是paddle的开源代码：https://github.com/minivision-ai/photo2cartoon-paddle   
也借鉴了paddle社区的优秀项目，动态剪影： https://aistudio.baidu.com/aistudio/projectdetail/764130 ，略修改了代码，将这个项目图片处理部分，用Image替代cv2，以便处理png等。   
过程：   
1.我们现将小组成员黄雨嫣的自拍视频，使用paddle的人体姿态检测，绑定宇航服骨骼，   
2.使用photo2cartoon-paddle项目，将成员脸部提取出，作为人物头像   
3.3.小组设计师彤彤设计了船票，留出png空白，将宇航员填入   

结果：
可以用单张图片输出静态船票：
![one](https://user-images.githubusercontent.com/70752098/112250589-98b55080-8c94-11eb-9226-c8301c898eeb.png)

可以用视频输出动态船票

