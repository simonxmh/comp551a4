# COMP 551 Project 4
Reproducing the paper "<strong>MULTI-TASK LEARNING ON MNIST IMAGE DATASETS</strong>".

# Usage
## multi.py
Run the code using the following command.
```python:multi.rb
python multi.py
```
<dl>
  <dt>What it does</dt>
  <dd>It constructs AllConv model [1] and runs multi-task training.</dd>
  <dt>The result</dt>
  <dd>Outputs files of the multi-task learning in text file format under the directory out.</dd>
</dl>

## p4_mnist.py
Run the code using the following command.
```python:multi.rb
python p4_mnist.py
```
<dl>
  <dt>What it does</dt>
  <dd>It constructs AllConv model [1] and runs single-task training.</dd>
  <dt>The result</dt>
  <dd>Outputs files of the sigle-task learning in text file format under the directory out.</dd>
</dl>

# Reference
* [1] Springenberg, Jost Tobias, et al. "Striving for simplicity: The all convolutional net." arXiv preprint arXiv:1412.6806 (2014).
* [2] Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
* [3] Argyriou, Andreas, Theodoros Evgeniou, and Massimiliano Pontil. "Multi-task feature learning." Advances in neural information processing systems. 2007.
