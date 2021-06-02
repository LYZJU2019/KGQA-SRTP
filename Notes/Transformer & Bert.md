Transformer有一个著名的应用叫做Bert

Transformer 就是SEQ2SEQ model加上self attention

![image-20210601104302524](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601104302524.png)

RNN（单项和双向）：输入是vector sequence，输出也是。单项RNN（输出是$b_i$，就会把$a_0,a_1,...,a_i$的输入看一遍）双向RNN（输出每一个$b_i$，都会把输入序列的每一个看一遍）。RNN适合处理输入是sequence的情况，但是不容易被平行化（以single direction为例，想要输出$b_4$，那么需要先看$a_1$，再$a_2$，最后是$a_4$。）于是想到用CNN代替RNN

![image-20210601105227901](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601105227901.png)

上图中，每一个三角形代表一个filter，这样的话每一个filter可以对序列中的特定部分进行计算，而且同一层的filter计算可以同时进行，每一种filter可以得到一个数值，换用不同的filter之后可以得到$b^1,b^2,b^3,b^4$这四个向量，然后再在此基础上叠CNN。这样就可以实现上层CNN对输入序列的全局做考察。（缺点就是需要叠很多层，计算耗时长）如果要求第一层就必须对全局考察，可以用self-attention。

![image-20210601105935537](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601105935537.png)

Self-attention可以做RNN可以做的一切事情，此外，$b^1,...,b^4$可以同时被计算出来。

![image-20210601110617938](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601110617938.png)

其中$W^i$为transformation matrices，$a^i,x^i,q^i,k^i,v^i$为向量，**拿每一个query去对每个key做attention**

![image-20210601111535580](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601111535580.png)

以$q^1$为例，$q^1$对每一个$k^i$做内积之后（可以理解为计算两个向量之间的相似度）除以$\sqrt{d}$（可以理解为消除向量维数对计算相似性的影响）

![image-20210601111904929](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601111904929.png)

再做一次softmax运算

![image-20210601112147644](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601112147644.png)

得到的$\hat\alpha_{1,1},...,\hat\alpha_{1,4}$再和$v^1,...,v^4$做点乘求和（相当于weighted sum）得到$b^1$，这相当于考虑了input sequence的全部（如果只想考虑local的话，就可以把不想考虑的部分的$\hat\alpha$设置为0）。

![image-20210601112743167](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601112743167.png)

同理$b^2,b^3,b^4$也可以同步算出来，这里同步计算可以用把向量拼接成矩阵来实现：

![image-20210601113227997](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601113227997.png)

（上图是通过不同的权重矩阵计算$q^i,k^i,v^i$）

![image-20210601113835247](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601113835247.png)

（上图通过逐个点乘计算$\alpha$，此处省略$\sqrt{d}$，得到的矩阵中每个元素表示输入的序列中两两之间的attention）

![image-20210601113921988](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601113921988.png)

（上图计算矩阵softmax）

![image-20210601114104710](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601114104710.png)

（weighted sum求出输出矩阵）

总结起来，就是一串矩阵乘法，可以使用GPU来加速：
$$
Q=W^q\cdot I\\
K=W^k\cdot I\\
V=W^v\cdot I\\
A=K^T\cdot Q\\
A\rarr\hat A\\
O=V\cdot \hat A
$$
此外还有一个变式，就是multi-head self-attention

![image-20210601115334743](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601115334743.png)

其中以$q$为例：
$$
q^i=W^qa^i\\
q^{i,1}=W^{q,1}q^i\\
q^{i,2}=W^{q,2}q^i
$$
$W^{q,i}$是事先设定好（训练出来的）权重矩阵。

![image-20210601115837295](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601115837295.png)

如上图所示，算出来的每一个query只会和对应的key做attention，得到的结果直接concatenate在一起。如果要满足矩阵维数上的要求，可以讲结果和输出权重矩阵$W^O$相乘，得到结果矩阵，如下图所示：

![image-20210601120042998](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601120042998.png)

好处：不同的head关注的点不一样（有的head用来关注局部的时序，有的head用来关注全局的序列，不同的head之间可以各司其职）

***想法：做智能对话系统的时候，可以用探测局部时序的head检测用户问句中的省略内容，用全局的head来对整个对话流程进行跟踪***

这个时候有一个很大的问题，那就是input sequence的顺序不影响输出（天涯若比邻），但这明显不是我们想要的，我们需要将输入的顺序信息考虑进self-attention模型中去。

![image-20210601125807374](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601125807374.png)

一篇论文提出positional encoding，每一个$a^i$算出来之后加上$e^i$（事先人工设定好的，不是通过训练得到的，象征着序列的位置信息）

![image-20210601130309207](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601130309207.png)

有一种解释，说明为什么是$e^i+a^i$，如上图所示，其中$p^i$是独热向量，$W^P$是认为设定的，有一个很奇怪的式子可以算出来，画出图像后，长这样：

![image-20210601130643363](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601130643363.png)

下面一个模型是Seq2seq with attention

![image-20210601131214015](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601131214015.png)

原始的模型encoder部分是双向的RNN，decoder部分是单向的RNN，现在全部换成了self-attention的layer

例子（使用该模型进行机器翻译）：

![image-20210601134035608](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601134035608.png)

其中前三行的encode的过程，两两之间做self attention，进行三次。后三行是decode的过程，除了将encoder产生的序列作为输入之外，还会将之前输出的$o^0,o^1,...,o^{i-1}$作为输出$o^i$的考虑中。

著名的transformer的架构（源自论文Attention is all you need）：

![image-20210601134552977](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601134552977.png)

Encoder部分：

![image-20210601135144032](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601135144032.png)

encoder部分讲解：输入序列先进过input embedding层（$x^i$乘以一个权重矩阵$W$后得到的$a^i$），通过positional encoding之后作为输入进入到中间灰色的block里面（会循环$n$次）。首先进入的是multi-head attention，得到序列$b^i$之后进入Add & Norm模块，此模块将multi-head attention模型的输入$a$和输出$b$相加，并进行layer normalization（具体参考文献https://arxiv.org/abs/1607.06450)。可以和batch normalization做类比，上图的右上角是一个batch normalization的实例，样本数据data按列摆放，$batch\_size=4$，我们希望一个batch里面不同data的相同dimension（batch的同一行是相同的dimension）的$mean=0,\sigma=1$。但是layer normalization是不需要考虑batch的，我们希望给定data，让data里面所有dimension的$mean=0,\sigma = 1$，一般layer normalization会搭配RNN一起使用。Feed forward结构（应该是一个fully connected network）会将前一部分的输出进行处理，同样有Add & Norm这一环节。（以上过程会重复$n$次，每次的参数都是不一样的）

decoder：输入的是前一个time step产生的output，经过positional encoding之后进入灰色的block（同样循环$n$次），然后进入Masked Multi-head Attention（所谓Masked，指的是模型仅仅专注于前面已经产生的序列，也就是说，未产生部分的$\alpha=0$），经过Add & Norm之后和encoder部分产生的序列同时进入multi-head attention，（中间省略一些叙述）之后进入linear模块，经过softmax函数之后得到输出的可能性向量。

![image-20210601142039670](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601142039670.png)

上图是原始paper中attention的结果，线条颜色越深表示关联度越大

![image-20210601142412838](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601142412838.png)

将transformer中的attention layer取出来分析之后的结果（自动attend到正确的位置）

![image-20210601143034959](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601143034959.png)

上图是multi-head attention的结果，可以看出，使用不同的query和key可以设计不同的attention head，达到不同的效果（上图上半部分的head探测的是global的信息，下半部分的head则更专注于local information）

基本上能用Seq2seq的，就可以使用transformer。

![image-20210601143718496](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601143718496.png)

Google创建了一个summarizer，input不是一篇文章，而是一个含有很多篇文章的set，output是一篇文章，具有Wikipedia的风格，（希望机器读完搜索引擎返回的文章之后写出一篇Wikipedia）

![image-20210601144109156](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601144109156.png)

在transformer发明之前，只能使用Seq2seq和RNN（测试样例无法多起来，否则模型会烂掉），但是有了transformer之后，测试样例就可以翻成千上万倍。（具体参考https://arxiv.org/abs/1801.10198)

还有一个transformer的变种叫做universal transformer，在深度上是一个RNN（在深度上copy $T$次，每次的参数都是一样的，广度上是position，具体参考（https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html)

![image-20210601144738438](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601144738438.png)

当初设计universal的动机：

![image-20210601160030053](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601160030053.png)

简单来说，就是想设计一个在翻译和算法人物表现都出色的transformer。

除此之外，universal transformer还有一个特殊的机制，称为dynamic halting，如下图所示：

![image-20210601163246546](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601163246546.png)

有的position不需要堆满（由model决定什么时候停下来）。在上图的例子中，position2需要的层数最多，于是当其他的position的层数超过所需的层数时，就会停下来，把数据作为position2的输入，让position2继续做self-attention。

测试和结果：

![image-20210601164600751](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601164600751.png)

这个dataset里面每个问题由context和target sentence组成，作为universal transformer的输入，transformer需要在context中找到一个词语（答案一定在context中），用来填target sentence的空缺处，使句意与context相同或者相近。

测试结果：

![image-20210601165031226](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601165031226.png)

结论：universal transformer（尤其是加上dynamic halting之后）的精确度优于普通的transformer；dynamic halting机制的引入可以显著降低模型的perplexity，使得模型有更好的generalization。

Transformer也可以用在图片影像中，比如下面的self-attention GAN

![image-20210601150004291](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601150004291.png)

（具体参考https://arxiv.org/abs/1805.08318)

**Fully connected network**

​	lots of model

**Convolutional neural network（可以抓住local的特征）**

​	ResNet

​	DenseNet (residual network)

​	Inception Network

**RNN（序列型data）**

​	Seq2seq

​	LSTM

​	GRU

​	Pointer Network

**Follow up SOTA structure!!!**

**why new architecture**

​	Increase performance

​	better feature extraction from data

​	generalization

​	reduce parameter and explainable

![image-20210601151936962](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601151936962.png)

![image-20210601152751449](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601152751449.png)

![image-20210601152448873](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601152448873.png)

Sandwich Transformer：通过对原有transformer的sublayer的顺序重新排列，设计一种更好的transformer。

![image-20210601153402296](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601153402296.png)

实验结论：self-attention layer叠起来放在靠近input的那一端，FCL叠起来放在靠近output的那一端，中间交错。这样的结构会有更好的效果。

![image-20210601153938268](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601153938268.png)

左上角的结果表明，这种reorder很不稳定，有些比baseline好

![image-20210601155348240](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601155348240.png)

进一步实验证明了上述结论

![image-20210601155436986](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601155436986.png)

![image-20210601155705720](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601155705720.png)

下一个架构名称叫做Residual Shuffle Exchange network

![image-20210601183317226](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601183317226.png)

![image-20210601183454892](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601183454892.png)

![image-20210601184002973](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601184002973.png)

shuffle可以取代attention取到远程的信息。

![image-20210601184339211](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601184339211.png)

![image-20210601184406403](C:\Users\22120\AppData\Roaming\Typora\typora-user-images\image-20210601184406403.png)



