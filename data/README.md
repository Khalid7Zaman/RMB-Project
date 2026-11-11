# Data

Face images are not part of this repository.

**FER2013** — public dataset (Goodfellow et al., 2015). Use the folder version with the official partition:

    data/fer2013/train/<class>/Training_<id>.jpg
    data/fer2013/test/<class>/{PublicTest,PrivateTest}_<id>.jpg

**DFERCD** — in-vehicle driver facial expression dataset collected by the authors. It contains identifiable faces
and is available from the corresponding author on reasonable request for non-commercial research:

    data/cmd/<Class>/<prefix> (<k>).jpg

The file-level split manifests used in the paper are in `outputs/manifests/` (paths relative to the repository
root). Rebuilding them with `python -m rmbff.data --fer2013 data/fer2013 --cmd data/cmd` gives identical splits.
