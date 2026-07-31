# Dataset audit

## Split totals

| split | images | annotations |
|---|---|---|
| train | 9752 | 95745 |
| valid | 2090 | 20602 |
| test | 2090 | 20536 |

## Leakage

- exact cross-split duplicate groups: **0**
- test images duplicating train/valid (dHash<=5 AND NCC>=0.98): **0**
- dHash candidates examined: 441240, of which rejected as look-alikes by pixel correlation: 441240
- within-split duplicate groups: {'train': 40, 'valid': 6, 'test': 11}

**VERDICT: LEAKAGE-FREE**

## Per-class instance counts

| class | group | train | valid | test |
|---|---|---|---|---|
| Filling | head | 34318 | 7359 | 7352 |
| impacted tooth | head | 19582 | 4204 | 4192 |
| Root Canal Treatment | head | 13386 | 2879 | 2866 |
| Crown | head | 7872 | 1693 | 1687 |
| Caries | head | 7505 | 1615 | 1604 |
| Periapical lesion | mid | 3695 | 799 | 797 |
| Missing teeth | mid | 2447 | 529 | 527 |
| Bone Loss | mid | 2188 | 473 | 469 |
| Root Piece | mid | 1824 | 397 | 392 |
| Implant | mid | 1251 | 270 | 270 |
| Mandibular Canal | mid | 433 | 96 | 92 |
| maxillary sinus | mid | 324 | 68 | 70 |
| post - core | mid | 219 | 50 | 46 |
| wire | mid | 161 | 35 | 35 |
| Primary teeth | mid | 157 | 37 | 34 |
| Retained root | mid | 113 | 31 | 30 |
| orthodontic brackets | tail | 94 | 21 | 21 |
| metal band | tail | 44 | 11 | 10 |
| Supra Eruption | tail | 32 | 8 | 8 |
| attrition | tail | 29 | 7 | 8 |
| abutment | tail | 24 | 5 | 4 |
| Malaligned | tail | 10 | 4 | 4 |
| Permanent Teeth | tail | 8 | 2 | 2 |
| Fracture teeth | tail | 7 | 2 | 2 |
| plating | tail | 6 | 0 | 2 |
| gingival former | tail | 4 | 2 | 8 |
| permanent retainer | tail | 4 | 2 | 2 |
| Cyst | tail | 3 | 1 | 1 |
| TAD | tail | 2 | 1 | 1 |
| Root resorption | tail | 2 | 1 | 0 |
| bone defect | tail | 1 | 0 | 0 |

## YOLO vs COCO reconciliation

- **train**: 9752/9752 images (coco/yolo), 95745/95745 annotations, 0 anns with empty segmentation, 0 classes disagree
- **valid**: 2090/2090 images (coco/yolo), 20602/20602 annotations, 0 anns with empty segmentation, 0 classes disagree
- **test**: 2090/2090 images (coco/yolo), 20536/20536 annotations, 0 anns with empty segmentation, 0 classes disagree

## Label problems

- train: {'coords_out_of_unit_range': 4, 'zero_area_polygon': 8}
- test: {'coords_out_of_unit_range': 2}
