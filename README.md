# DML-for-Classification [🚧 WIP]

This series of experiments is to curb my curiosity on:
1. How Deep Metric Learning or distance based methods perform on classification tasks?
2. Does generalization to unseen classes holds true?
3. Why is Deep Metric Learning that performs exceedingly well for biometrics (Face recognition), isn't used against other image recognition problems?
4. Are the trade offs worth choosing a distance based approch over a classical classification approach?

## Motivation

It all started with a company research project, when my [Manager](https://www.linkedin.com/in/vikascm?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BdCInhNbuTaquKbrE39Ll4Q%3D%3D) asked me to take up classification problem but approach in a more of a class agnostic way, in general sense try embedding / DML / distance based methods. Developing a model in this way has certain advantages like, we can add <b> more data and classes </b>without retraining everytime.This approach seemed promising, considering that established <b>Face recognition technology </b>demonstrates its feasibility, yet its widespread adoption remains limited.With the advent of Vector Databases, I believe we will more seeing towards this shift. 

## Experiment Comparison & Tracking
I'm using Weights & Biases MLops tool to track and compare all experiments. The goal is to stick to predefined metrics and not change over the course of experimentation to give apples to apples comparison.<br/>  [Link to W&B experimentation page](https://wandb.ai/pranavjadhav001/embedding_based_classification?nw=nwuserpranavjadhav001)

## Dataset
Decent amount of time was spent on choosing the right dataset. I was looking at datasets particularly used for fine grained image classification problems with decent intra and inter class variance. I also wanted dataset with decent number of classes since unseen class validation split would reduce training classes. It has been known that having large no. of classes during training improves model generalizes capabilities and really brings out the best candidate for loss metric. CUB-200-2011 Dataset ticked all the above requirements. It has around 200 classes with approx 60 images for each class. It is commonly used to quantify DML tasks. Other datasets considered were : Cars196, Standford Online Products, In-shop Clothes retrieval , Hotel-50k.<br/>
Dataset [Link](https://paperswithcode.com/dataset/cub-200-2011)<br/>
### Download Dataset using CLI
```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
tar -xvzf <zip-filename>
```
## How to Quantify

There are various metrics to track and compare experiments involving Deep Metric Learning:
- NMI
- Recall @ k
- Mean average Precision(MAP)

Since the goal of this project is to check whether DML can replace traditional classification methods. I've chosen <b>precision @ 1</b> which literally translates to [classification accuracy](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/249).
## Common Experiment Details
All experiments followed these details, unless explicitly mentioned:
- Epochs 100
- Batch size 128
- fixed seed 42
- image shape 224x224x3
- image train test split 0.2
- unseen class train test split, last 20 classes from 180-200 were kept for unseen class metrics
- metric for test accuracy precision@1
- Augmentation is consistent wherever mentioned:
    - Train Transform
        - Resize 224x224x3
        - Random Crop 224x224x3
        - Random Horizontal Flip
    - Test Transform
        - Resize 224x224x3
        - Center Crop 224x224x3

## Experiment Table
<table class="">
   <thead>
      <tr id="d5c49a58-db50-4f0a-9038-b29cd9fcd0fe">
         <th id="ss;p" ><sub><strong>Experiment Name</strong></th>
         <th id="S]C:" ><sub><strong>Experiment Details </strong></th>
         <th id="`UGS" ><sub><strong>Deductions</strong></th>
      </tr>
   </thead>
   <tbody>
      <tr id="ee31b7f0-d1e2-4899-ac2d-e76bbc631317">
         <td id="ss;p" class="" ><sub>EPSHN_euclidean_resnet18</sub></td>
         <td id="S]C:" class="" ><sub>- Euclidean Distance metric<br/>- used <a href="#references">EPSHN</a> as loss <br/>- Architecture chosen as Resnet 18 with Imagenet1k Pretrained weights<br/>- Used Adam optimizer<br/>- Embedding Dimension of 128<br/><br/></sub></td>
         <td id="`UGS" class="" ><sub>- Test Accuracy of 50 % was achieved </sub></td>
      </tr>
      <tr id="b0b47476-297c-4387-a27c-fe30d9f64d1b">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_euclidean_resnet18<br/>- used cosine as distance metric<br/></sub></td>
         <td id="`UGS" class="" ><sub>- Loss and accuray curves followed  EPSHN_euclidean_resnet18<br/>- slight test accuracy gain of 2% w.r.t EPSHN_euclidean_resnet18 <br/></sub></td>
      </tr>
      <tr id="077f1041-4613-43d3-a235-e712d033c05b">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18_sgd</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18<br/>- used SGD optimizer instead of adam<br/><br/></sub></td>
         <td id="`UGS" class="" ><sub>- around 25% gain in test accuracy<br/>- SGD generalizes really well, making model more robust towards unknown data distribution and classes<br/></sub></td>
      </tr>
      <tr id="0e0b3293-da57-42c0-b9e4-a8950ece422c">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd<br/>- added train and test augmentations for generalization<br/></sub></td>
         <td id="`UGS" class="" ><sub>- around 3% test accuracy gains<br/>- This will be treated as Baseline<br/>- a drop of 20% precision/accuracy was noted for model performance b/w with and without classes included<br/></sub></td>
      </tr>
      <tr id="ecb11b81-2a59-4fe7-881f-f9521f83e8f6">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_frozenBN_resnet18_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- all batch normalization layers are frozen , and pretrained imagenet1k parameters are retained<br/></sub></td>
         <td id="`UGS" class="" ><sub>- Frozen BN is usually to reduce overfitting and generalize better to unknown data and classes<br/>- this step led to decrease in test accuracy by around 14% w.r.t to baseline<br/></sub></td>
      </tr>
      <tr id="1ea7ce2a-46a7-4d80-b02f-52cbae080892">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18_scratch_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- instead of pretrained weights, model was trained from scratch<br/></sub></td>
         <td id="`UGS" class="" ><sub>- model couldn’t even reach test accuracy of 20%<br/>- This tells weight initialization plays a crucial role in convergence of the model<br/></sub></td>
      </tr>
      <tr id="987abe54-e489-4390-ad3f-f10b6826fd60">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18_classifier_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- instead of pretrained weights, model was first trained using classifier and later finetuned using Metric learning<br/></sub></td>
         <td id="`UGS" class="" ><sub>- model oscillates over the same point in loss and accuracy curve which it inherits from pretrained weights from the classifier<br/>- test accuracy will depend how well classifier is trained , and how well it generalizes<br/></sub></td>
      </tr>
      <tr id="61990a8f-0a1f-4426-90f7-3740f7703ac9">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18x512_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- Embedding Dimension of 512 was chosen instead of 128<br/></sub></td>
         <td id="`UGS" class="" ><sub>- Increasing embedding dim doesn’t have effect on test accuracy<br/>- only a drop of 10% precision/accuracy was noted for model performance b/w with and without classes included<br/>- for unseen class performs better than baseline<br/></sub></td>
      </tr>
      <tr id="53b6f915-85a2-462d-b873-f31b54b62e7f">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18+skipConnHead_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>-  same as EPSHN_cosine_resnet18_sgd_aug<br/>- Adding Skip connection Head on top the model<br/></sub></td>
         <td id="`UGS" class="" ><sub>- Adding skip connection head doesn’t have effect on test accuracy</sub></td>
      </tr>
      <tr id="6d42ccf5-3933-452b-a3b2-31ccc55a3010">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet18+1dBN_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- Adding a 1d Batch Normalization on top of model<br/></sub></td>
         <td id="`UGS" class="" ><sub>- Adding 1D Batch Norm layer doesn’t have effect on test accuracy<br/>- only a drop of 10% precision/accuracy was noted for model performance b/w with and without classes included<br/>- for unseen class performs better than baseline<br/></sub></td>
      </tr>
      <tr id="05045bd3-5851-458e-a254-c881a7979bca">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_resnet50_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- instead of Resnet 18 , Resnet 50 was chosen with pretrained imagenet1k weights<br/></sub></td>
         <td id="`UGS" class="" ><sub>- Adding more layers / parameters has a direct correlation on performance<br/>- an increase of 11% test accuracy was noted w.r.t baseline<br/>- only a drop of 10% precision/accuracy was noted for model performance b/w with and without classes included<br/></sub></td>
      </tr>
      <tr id="807ca97a-d35b-438f-8400-e6d638c481d4">
         <td id="ss;p" class="" ><sub>ProxyNCA_cosine_resnet18_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- instead of EPSHN triplet loss, a proxy based loss  was used <br/>- better part of this method, it doesnt require any mining methods.<br/>- but you are also training additional parameters in the form of proxies<br/></sub></td>
         <td id="`UGS" class="" ><sub>- a huge drop in test accuracy around 16% w.r.t baseline</sub></td>
      </tr>
      <tr id="c684dc34-9dcf-4d7c-aa93-90934fa13e29">
         <td id="ss;p" class="" ><sub>ArcFace_cosine_resnet18_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- instead of EPSHN triplet loss, arcface loss was used <br/>- better part of this method, it doesnt require any mining methods.<br/>- but you are also training additional parameters in the form of class weights<br/></sub></td>
         <td id="`UGS" class="" ><sub>- a increase of 11 % test accuracy was noted from baseline<br/>- Arcface shows the highest drop (32%) in precision for model performance b/w with and without classes included<br/>- Model doesn’t generalize well for unseen classes <br/></sub></td>
      </tr>
      <tr id="acf38427-49c1-4821-a9a5-eb1a6f3ecc04">
         <td id="ss;p" class="" ><sub>SubCenterArcFace_cosine_resnet18_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as ArcFace_cosine_resnet18_sgd_aug<br/>- a variation of arcface loss used to help with datasets with high intra class variance <br/></sub></td>
         <td id="`UGS" class="" ><sub>-  a increase of 1 % test accuracy as compared to baseline<br/>- Exp takes up after ArcFace_cosine_resnet18_sgd_aug on generalization for unseen classes<br/></sub></td>
      </tr>
      <tr id="0c51bc13-6081-4407-bdc2-3428a0997c32">
         <td id="ss;p" class="" ><sub>EPSHN_cosine_simclr_resnet18_sgd_aug</sub></td>
         <td id="S]C:" class="" ><sub>- same as EPSHN_cosine_resnet18_sgd_aug<br/>- instead of pretrained imagenet1k weights, model used weights learned from simclr model trained on CUB-200-2011 Dataste <br/></sub></td>
         <td id="`UGS" class="" ><sub>- model couldn’t even reach test accuracy of 20%<br/>- weight initialization plays a crucial role in convergence of the model<br/></sub></td>
      </tr>
   </tbody>
</table>


## How an experiment has been conducted
1. Create experiment config , initialize wandb run
2. Initialize random seed to a fixed value which is common for all experiments 
3. Load image dataset for all 200 classes , split into train test datasets with 0.2 ratio stratified at label level
4. Initialize model, dataloaders, optimizer, distance metric, loss function, validation function
5. Train for 100 epochs while monitoring train loss and test accuracy curves, save the weights at the end after training
6. For test dataset ,get precision @ 1 scores for all classes,for each test sample it looks for closest sample in training dataset; "all_classes_metrics" Table
7. Using Faiss library, For each test sample , get predicted class by searching for closest sample in training dataset, Generate a classification report ; "all_classes_classification_report" Table
8. Train a new model, excluding last 20 classes from dataset.
9. All the steps(4,5,6,7) remain the same as first phase training, even during evaluation, entire dataset was loaded including the last 20 classes 
10. Get precision drop for last 20 unseen classes for both the models; call it "comparison_unseen_classes_metrics" Table, average leads to "precision_drop_unseen_classes" scalar.
11. Get precision drop for all classes for both the models; call it "comparison_seen_classes_metrics" Table, average leads to "precision_drop_seen_classes" scalar.

## What to look for
More comprehensive plots and numbers were created for each experiment to better compare against each other.These plots, tables and scalars are only present for experiments that showed potential(decent test accuracy)
- <b>"train_loss" curve</b>: Tracks the loss curve for 200 class training Dataset 
- <b>"test_accuracy" curve</b> : Tracks precision @ 1 curve for the 200 known classes for test dataset
- <b>"all_classes_metrics" table</b> : Table which contains rows of class name, test sample no. , training sample no. , precision @ 1 for that class. Determined by looping for all classes, for each test sample find the class of the closest sample in training dataset. 
- <b>"all_classes_classification_report" table </b>: classification report table for test dataset. 
- <b>"train_loss2" curve</b>: Tracks the loss curve for 180 class training Dataset
- <b>"test_accuracy2" curve</b>: Tracks precision @ 1 curve for the 180 known classes for test dataset
- <b>"limited_classes_metrics" table</b> : Table which contains rows of class name, test sample no. , training sample no. , precision @ 1 for that class. Determined by looping for all 200 classes, for each test sample find the class of the closest sample in training dataset. 
- <b>"limited_classes_classification_report" table</b> : classification report table for test dataset for all 200 classes
- <b>"comparison_unseen_classes_metrics" table</b> : table which contains rows of class name, precision @ 1 for 200 class model, precision @ 1 for 180 class model
- <b>"precision_drop_unseen_classes" scalar</b> : average value of unseen 20 classes(precision @ 1 for 200 class model  - precision @ 1 for 180 class model)   
- <b>"precision_drop_all_classes" scalar</b> : average value of 200 classes(precision @ 1 for 200 class model  - precision @ 1 for 180 class model)

## Explainability
- To understand and see where the model is looking at, implemented Grad Cam equivalent for embedding networks from this [paper](https://arxiv.org/abs/2402.00909). You can see the results here at this [repo](https://github.com/pranavjadhav001/embedding-grad-cam-pytorch). Model from <b>EPSHN_cosine_resnet50_sgd_aug</b> experiment has been used.
- To visualize embeddings for all classes for entire dataset, I've used [FiftyOne](https://docs.voxel51.com/tutorials/image_embeddings.html?highlight=embeddings) to plot using umap and see corresponding images associated. It shows how visually similar classes cluster around one another. This also tells us about the spread(variance) of test embeddings vs train embeddings. 
- Used [FAISS](https://github.com/facebookresearch/faiss) to find best and closest matches for unseen classes. This offers insights that closest one may not be always the best and hence always determine the best class by majority voting for closest <b>n samples</b>.

## References
- [Resnet-18 1D batchnorm Arcface architecture](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/resnet.py)
- ArcFace [Paper](https://arxiv.org/abs/1801.07698)
- EPSHN [Paper](https://www.notion.so/Improved-Embeddings-with-Easy-Positive-Triplet-Mining-50a8fa57c8b8462fbeb61160594a04eb?pvs=4)
- [Evaluation Measures information retreival](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#R-precision)
- Pytorch Metric Learning [Library](https://kevinmusgrave.github.io/pytorch-metric-learning/)
- [what-is-the-mean-average-precision-in-information-retrieval](https://www.educative.io/answers/what-is-the-mean-average-precision-in-information-retrieval)
- [precision-and-recall-in-information-retrieval](https://jamesmccaffrey.wordpress.com/2016/10/24/precision-and-recall-in-information-retrieval/)
- Pytorch Metric Learning [Library](https://kevinmusgrave.github.io/pytorch-metric-learning/)
- Quarterion Similarity Learning [Tips & Tricks](https://quaterion.qdrant.tech/tutorials/cars-tutorial.html)
- ProxyNCA Loss [Paper](https://arxiv.org/pdf/1703.07464)
- SimCLR pytorch implementation by [AiSummer](https://theaisummer.com/simclr/)
- A Metric Learning Reality Check [Paper](https://arxiv.org/pdf/2003.08505v3.pdf)
- Deep Metric Learning Survey [Blog Post](https://hav4ik.github.io/articles/deep-metric-learning-survey)

