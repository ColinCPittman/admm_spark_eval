## Disclaimer
Python 3 with pip may be needed.

The files are too large and we may not have the rights to host them, but you can install them locally to your machine here, they should be ignored by the .gitignore if the names aren't changed. Just make sure you're only staging what we need for the commits. 

## Download Instructions
If you're on Windows, you may be able to use ```download_rcv1.bat``` and ```download_higgs.bat``` to get the datasets. They will download files to the parent directory when run, which will be the /sourced folder above this one.

I got Higgs from [Kaggle](https://www.kaggle.com/datasets/erikbiswas/higgs-uci-dataset) initially, which required a free account to be setup.
The script may expect an API token: kaggle.com -> Account -> API -> Create New API Token. Save kaggle.json to %USERPROFILE%\.kaggle\
Note: API token may not be necessary if already logged into Kaggle in browser, I did not need to create an API token myself. 

RCV1 can be found manually [here](https://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/). If the script worked, you'll need [software like 7zip to extract them.](https://www.7-zip.org/). If that is not your style then:

Alternatively, they can be found in our private Teams group files in the Datasets folder. You can download them there and save to the parent folder (data/sourced/).





