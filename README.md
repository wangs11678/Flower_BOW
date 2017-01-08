# Flower_BOW
云南花卉识别项目，BOW算法。
/*
* Flower Classification
* 方法：Bag of Words
* 步骤描述
* 1. 提取训练集中图片的feature。
* 2. 将这些feature聚成n类。这n类中的每一类就相当于是图片的“单词”，
*    所有的n个类别构成“词汇表”。本文中n取1000，如果训练集很大，应增大取值。
* 3. 对训练集中的图片构造Bag of Words，就是将图片中的feature归到不同的类中，
*    然后统计每一类的feature的频率。这相当于统计一个文本中每一个单词出现的频率。
* 4. 训练一个多类分类器，将每张图片的Bag of Words作为feature vector，
*    将该张图片的类别作为label。
* 5. 对于未知类别的图片，计算它的Bag of Words，使用训练的分类器进行分类。
*/
