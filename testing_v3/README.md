# 复现

离轴K_l/K_r与pose_l/pose_r全部为0，由考生填写；填写后再校验和前向。以下命令为当前100个Lego输入生成待填写模板。

~~~bash
python code/inference.py --index ../../test_input/lego/test.json --out my_template --group group_example --format-demo
~~~
