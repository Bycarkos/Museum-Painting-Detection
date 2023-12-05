
# Museum Painting Detection

This Project is a custom dessigned CBIR for Museum painting retrieval. In this project we appy classical computer vision algorithms and tecnics in order to solve a common problem nowadays.

# Structure
```
├── data
├── src
│   ├── configs
│   │   ├── descriptors
│   │   ├── evaluation
│   │   └── preprocessing
│   ├── core
│   ├── descriptors
│   ├── methods
│   │   
│   ├── preprocessing
│   │   
│   └── utils
│       
└── tools
    ├── evaluation
    └── utils

```


# Configuration

in order to configure the environment

```bash
bash env.sh

```

The CBIR system has been done with hydra configuration system. 

The whole project can be modyfied from the run.yaml within the src/configs.

the idea is the following:

```
defaults:
  - _self_
  - descriptors: [default]
  - preprocessing: [background_removal, text_removal]
  - evaluation: [default]

data:

  BBDD:
    path: ${hydra:runtime.cwd}/data/BBDD
    importation:

      descriptors:
        import_: False
        path: ${hydra:runtime.cwd}/data/descritptors/hogpyramial.pkl


    export:

      descriptors:
        save: True
        path: ${hydra:runtime.cwd}/data/descritptors/hogpyramial.pkl

  QN: qsd2_w3


  QS:
    path: ${hydra:runtime.cwd}/data/${data.QN}

    preprocessed:
      import_: True
      export_: True
```

Here you have the main configuration to replicate the numbers, it is defined by default.

the deafults are the config files for the preprocessing and the computation of the descriptos and what evaluate, you can check the configuration there.

in order to change something you can do it by command line, however i strngly recommend to chek the hydra documentation first: https://hydra.cc/docs/intro/ .

## Run the script

from root of the project just

```python
python src/main

```

if you have already preprocessed the database and get the descriptors you can to rerun the script by:

```python
python src/main.py data.BBDD.importation.import_=True data.QS.preprocessed.importation=True

```


## Collaborators

Carlos Boned Riera

Iker García Fernández

Goio García Moro

Xavier Micó Pérez
