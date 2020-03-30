# Morphological tessellation as a way of partitioning space
Code and data repository for **Morphological tessellation as a way of partitioning space: Improving consistency in urban morphology at the plot scale**.

Fleischmann, M., Feliciotti, A., Romice, O. and Porta, S. (2020) _‘Morphological tessellation as a way of partitioning space: Improving consistency in urban morphology at the plot scale’_, Computers, Environment and Urban Systems, 80, p. 101441. doi: [10.1016/j.compenvurbsys.2019.101441](http://doi.org/10.1016/j.compenvurbsys.2019.101441).

Archived version of this repository is stored at the University of Strathclyde KnowledgeBase at DOI [10.15129/c766db26-3fa8-45c6-8218-098d529571fc](https://doi.org/10.15129/c766db26-3fa8-45c6-8218-098d529571fc).

Contact: martin@martinfleischmann.net

Date: 29/03/2020

## Abstract
Urban Morphometrics (UMM) is an expanding area of urban studies that aims at representing and measuring objectively the physical form of cities to support evidence-based research. An essential step in its development is the identification of a suitable spatial unit of analysis, where suitability is determined by its degree of reliability, universality, accessibility and significance in capturing essential urban form patterns. In Urban Morphology such unit is found in the plot, a fundamental component in the morphogenetic of urban settlements. However, the plot is a conceptually and analytically ambiguous concept and a kind of spatial information often unavailable or inconsistently represented across geographies, issues that limit its reliability and universality and hence its suitability for Urban Morphometric applications. This calls for alternative methods of deriving a spatial unit able to convey reliable plot-scale information, possibly comparable with that provided by plots.

This paper presents Morphological Tessellation (MT), an objectively and universally applicable method that derives a spatial unit named Morphological Cell (MC) from widely available data on building footprint only and tests its informational value as proxy data in capturing plot-scale spatial properties of urban form. Using the city of Zurich (CH) as case study we compare MT to the cadastral layer on a selection of morphometric characters capturing different geometrical and configurational properties of urban form, to test the degree of informational similarity between MT and cadastral plots.

Findings suggest that MT can be considered an efficient informational proxy for cadastral plots for many of the tested morphometric characters, that there are kinds of plot-scale information only plots can provide, as well as kinds only morphological tessellation can provide. Overall, there appears to be clear scope for application of MT as fundamental spatial unit of analysis in Urban Morphometrics, opening the way to large-scale urban morphometric analysis.

## Repository
This repository contains Python code used wihtin the research. Few minor steps were done in QGIS 3 and ArcMap 10.6, some figures were post-processed in Adobe Illustrator. This repository does not contain complete data due to the size limitations. **Complete geospatial data are openly available from the University of Strathclyde KnowledgeBase at DOI [10.15129/c766db26-3fa8-45c6-8218-098d529571fc](https://doi.org/10.15129/c766db26-3fa8-45c6-8218-098d529571fc).** together with an archived version of this repository.

Python code is stored wihtin Jupyter notebooks. For the accessibility purposes, contents of notebooks were also exported into executable scripts and PDF. 

Data folder contains all data resulting from analysis described in the paper are stored in CSV.

Note: notebooks has been cleaned and released retroactively. It is likely that different versions of packages were initially used, but we made sure that the results remained unaltered.


## Data structure
```
*.ipynb
    - Jupyter notebooks
environment.yml
    - conda environment specification to re-create reproducible Python environment
data/
    args_test.gpkg - data for parameter tests
    areas.csv - area values from parameter tests
    perimeters.csv - perimeter values from parameter tests
    points.csv - point count values from parameter tests
    times.csv - timer values from parameter tests
    Results_*.csv - results of comparitive analysis
    single_uids.csv - IDs of buildings being alone on a single plot (QGIS generated)
    zurich.gpkg - input building data for tessellation (available at UoS KnowledgeBase)
    contiguity_diagram.gpkg - samples to be used in contiguity diagram
    
    cadastre/ (available at UoS KnowledgeBase)
        blg_cadvals.shp - cadastral values spatially joined to buildings (available at UoS KnowledgeBase)
        cadastre.shp - processed cadastral layer (available at UoS KnowledgeBase)
        Zurich_cadastre.shp - input cadastral layers (available at UoS KnowledgeBase)
        
    network/ (available at UoS KnowledgeBase)
        network.shp - street network data (available at UoS KnowledgeBase)
        + additional data used in ArcMap 10.6 UNA Toolkit (available at UoS KnowledgeBase)
        
    tessellation/ (available at UoS KnowledgeBase)
        {k}_tessellation.shp - tessellation layers (available at UoS KnowledgeBase)
```

## License

Contents of this repository is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

Source data: Vektor-Übersichtsplan des Kantons Zürich, 13.03.2018, Amt für Raumentwicklung Geoinformation / GIS-Produkte, Kanton Zürich, https://opendata.swiss/de/dataset/vektor-ubersichtsplan1