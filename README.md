Fork from abdulfatir/twitter-sentiment-analysis. 
**本项目主要学习lstm对Twitter文本进行分类. **

- 补充了train/test.csv文件, 下载自[Link](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/), shuffle后, 节选了10, 后按照20/80的比例划分train/test.
- 新建summary.ipynb文件, 综合了preprocess, stats和lstm文件, 可以直接运行
- 在summary中, 对下列文件进行了详细的中文注释, 并稍作修改:
 - preprocess.py
 - stats.py
 - lstm.py

- 就summary部分, 需要的package
 - numpy
 - nltk
 - keras
 - h5py
- 其余程序依赖包可参见README_original.md
