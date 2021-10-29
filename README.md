<h2>Reproduce Submit</h2>
<hr>

<h3>1. Enviornment</h3>
<h4>python version:</h4>
3.7.11<br>
<h4>package:</h4>
pytorch=1.7.1<br>
torchvision=0.8.2<br>
pillow=8.4.0<br>
pytorch_pretrained_vit=0.0.7<br>
timm=0.4.12<br>
einops=0.3.2<br>

<h3>2. Download Reproduce Pack</h3>
<a href="https://drive.google.com/file/d/1fSYY7UpaJAvBWap7LwjhOMMhzjLfIfvw/view?usp=sharing">
reproduce zip pack
</a>

<h3>3. Reproduce</h3>
a. unzip the download pack<br>

b. edit the **inference.py**<br> 
Line 11
> device = torch.device('cpu') # default, if your device doesn't have gpu <br>
> device = torch.device('cuda:0') # if your device have gpu

Line 17: assign where is test image folder<br>
notice: MUST end up with '/'
> test_folder = '' # default is empty<br>

example:
> test_folder = '../dataset/testing_images/'<br>

c. run inference.py

d. answer.txt will be produce