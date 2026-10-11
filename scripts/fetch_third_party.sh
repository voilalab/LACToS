#!/bin/bash
# Clone the official baseline implementations at the commits used for Table 1 into third_party/.
# PatchCore, SimpleNet, GLASS and SuperSimpleNet are imported from these clones by baselines/infer_feature.py.
# (DDAD and EfficientAD need no upstream code: DDAD's modules are in baselines/ddad_lib/, EfficientAD's
# checkpoints are pickled nn.Modules.)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT/third_party"
fetch() {  # name url commit
  if [ ! -d "$1" ]; then git clone -q "$2" "$1"; fi
  (cd "$1" && git fetch -q && git checkout -q "$3" && echo "$1 @ $(git rev-parse --short HEAD)")
}
fetch patchcore-inspection https://github.com/amazon-science/patchcore-inspection.git fcaa92f
fetch SimpleNet            https://github.com/DonaldRR/SimpleNet.git                 351a2b8
fetch SuperSimpleNet       https://github.com/blaz-r/SuperSimpleNet.git              98ab4d5
fetch GLASS                https://github.com/cqylunlun/GLASS.git                    f788d56
