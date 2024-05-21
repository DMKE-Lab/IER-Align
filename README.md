# IER-Align
IER-Align: Cross-Lingual Temporal Knowledge Graph Entity Alignment Model Based on Interactive Entity-Relation Alignment Network

## Datasets
ent_ids_1: ids for entities in source KG;

ent_ids_2: ids for entities in target KG;

ref_ent_ids: entity alignments

triples_1: relation quadruples encoded by ids in source KG;

triples_2: relation quadruples encoded by ids in target KG;

## Environment

```
Anaconda>=4.5.11
Python>=3.7.11
pytorch>=1.10.1
```

## Running

```
python train.py --data data/DBP15K --lang zh_en
```
