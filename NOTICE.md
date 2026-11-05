# Notice

## TrashNet dataset

This project trains and evaluates its models on the TrashNet dataset by Gary Thung and Mindy Yang, collected for their Stanford CS 229 (Machine Learning) project in 2016.

- Repository: https://github.com/garythung/trashnet
- Dataset download: https://huggingface.co/datasets/garythung/trashnet
- Report: Gary Thung and Mindy Yang, "Classification of Trash for Recyclability Status", CS 229 project report, Stanford University, 2016. https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf

The dataset authors ask everyone who uses the dataset to cite their repository:

> Gary Thung and Mindy Yang. trashnet. https://github.com/garythung/trashnet

The dataset images are not included in this repository. They are downloaded separately.

TrashNet is distributed under the MIT License:

```
MIT License

Copyright (c) 2017 Gary Thung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## RealWaste dataset

The real-world check in [docs/evaluation.md](docs/evaluation.md#real-world-photos-and-robustness) uses the RealWaste dataset by S. Single, S. Iranmanesh and R. Raad: photos of waste items received at the Whyte's Gully Waste and Resource Recovery facility in Wollongong, Australia.

- Repository: https://github.com/sam-single/realwaste (version used: commit 156e8e9154eeb163df30fcfd767f55b19633bb06)
- Paper: S. Single, S. Iranmanesh and R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning", Information 14(12), 633, 2023. https://doi.org/10.3390/info14120633

The authors ask everyone who uses the dataset to cite this paper. The repository states that RealWaste is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence (CC BY-NC-SA 4.0), https://creativecommons.org/licenses/by-nc-sa/4.0/. Its images are not included in this repository and are downloaded separately. Only measured results (metrics and a confusion matrix) are published here.
