"""BoneVision Loader: open a BoneVision femur reconstruction folder in 3D Slicer.

Use it in any of three ways:

* As a module: Edit > Application Settings > Modules > Additional module paths,
  add this ``BoneVisionLoader`` folder, restart Slicer, then pick
  Modules > BoneVision > BoneVision Loader.
* From a terminal: ``Slicer --python-script BoneVisionLoader.py <output_folder> [orientation]``.
  Inside an unzipped web-app bundle the folder argument can be left out.
* From Slicer's Python console:
  ``exec(open("/path/BoneVisionLoader.py").read()); load_bonevision_result("/path/output")``

The folder is what ``scripts/reconstruct_single_xray.py`` (or the web app's
Slicer bundle) writes: ``femur_reconstruction.stl``, ``femur_labelmap.nrrd``
and ``reconstruction.json``. Older folders that only contain the STL also load.
"""
import json
import os
import sys

import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

ORIENTATIONS = {
    "upright_right_femur": "Upright, right femur (view only)",
    "upright_left_femur": "Upright, left femur (view only)",
    "canonical": "Canonical SSM frame",
}
# Kept in sync with scripts/slicer_export.py; used when a report predates it.
FALLBACK_TRANSFORMS = {
    "canonical": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "upright_right_femur": [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    "upright_left_femur": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
}
BONE_COLOR = (0.89, 0.85, 0.72)


def load_bonevision_result(folder, orientation="upright_right_femur"):
    """Load one reconstruction folder into the scene and return the created nodes."""
    folder = os.path.abspath(os.path.expanduser(folder))
    report_path = os.path.join(folder, "reconstruction.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    info = report.get("slicer", {})
    mesh_path = os.path.join(folder, info.get("mesh", report.get("output_mesh", "femur_reconstruction.stl")))
    label_path = os.path.join(folder, info.get("labelmap", "femur_labelmap.nrrd"))
    if not os.path.exists(mesh_path) and not os.path.exists(label_path):
        raise ValueError("No BoneVision outputs found in " + folder)
    name = os.path.basename(folder.rstrip(os.sep)) or "BoneVision"

    nodes = {}
    matrix = info.get("display_transforms", FALLBACK_TRANSFORMS).get(orientation)
    if matrix is None:
        raise ValueError("Unknown orientation " + repr(orientation) + "; choose one of " + ", ".join(ORIENTATIONS))
    transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name + " display orientation")
    vtk_matrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vtk_matrix.SetElement(r, c, matrix[r][c])
    transform.SetMatrixTransformToParent(vtk_matrix)
    nodes["transform"] = transform

    if os.path.exists(mesh_path):
        # The exporter writes canonical millimetres as RAS; forcing RAS also keeps
        # older headerless STLs from being flipped as LPS.
        model = slicer.util.loadModel(mesh_path, properties={"coordinateSystem": "RAS"})
        model.SetName(name + " femur surface")
        display = model.GetDisplayNode()
        display.SetColor(*BONE_COLOR)
        display.SetSliceIntersectionVisibility(True)
        model.SetAndObserveTransformNodeID(transform.GetID())
        nodes["model"] = model

    if os.path.exists(label_path):
        labelmap = slicer.util.loadLabelVolume(label_path, properties={"show": False})
        labelmap.SetName(name + " femur labelmap")
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name + " femur segmentation")
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(labelmap)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentation)
        seg = segmentation.GetSegmentation()
        if seg.GetNumberOfSegments():
            segment = seg.GetNthSegment(0)
            segment.SetName("femur estimate")
            segment.SetColor(*BONE_COLOR)
        # The STL already shows the surface in 3D; the segmentation is for slice views and editing.
        segmentation.GetDisplayNode().SetVisibility3D(False)
        for node in (labelmap, segmentation):
            node.SetAndObserveTransformNodeID(transform.GetID())
        nodes["labelmap"], nodes["segmentation"] = labelmap, segmentation
        slicer.util.setSliceViewerLayers(background=None, label=labelmap, fit=True)

    if report:
        text = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", name + " report")
        text.SetText(json.dumps(report, indent=2))
        nodes["report"] = text
        warning = report.get("warning")
        if warning:
            print("BoneVision: " + warning)

    layout = slicer.app.layoutManager()
    if layout is not None:
        view = layout.threeDWidget(0).threeDView()
        view.resetFocalPoint()
        view.resetCamera()
    return nodes


class BoneVisionLoader(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "BoneVision Loader"
        parent.categories = ["BoneVision"]
        parent.contributors = ["BoneVision project"]
        parent.helpText = (
            "Loads a BoneVision femur reconstruction folder: surface model, labelmap as a segmentation, "
            "and the run report. Display orientations are rotations for viewing only; the estimate is "
            "not registered to patient anatomy. Research use only."
        )
        parent.acknowledgementText = ""


class BoneVisionLoaderWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        import ctk
        import qt

        form = qt.QFormLayout()
        self.folderButton = ctk.ctkDirectoryButton()
        form.addRow("Output folder:", self.folderButton)
        self.orientationBox = qt.QComboBox()
        for key, label in ORIENTATIONS.items():
            self.orientationBox.addItem(label, key)
        form.addRow("Orientation:", self.orientationBox)
        self.loadButton = qt.QPushButton("Load reconstruction")
        form.addRow(self.loadButton)
        self.layout.addLayout(form)
        self.layout.addStretch(1)
        self.loadButton.connect("clicked(bool)", self.onLoad)

    def onLoad(self):
        try:
            key = self.orientationBox.itemData(self.orientationBox.currentIndex)
            BoneVisionLoaderLogic().load(self.folderButton.directory, key)
        except Exception as exc:
            slicer.util.errorDisplay("Could not load BoneVision result: " + str(exc))


class BoneVisionLoaderLogic(ScriptedLoadableModuleLogic):
    def load(self, folder, orientation="upright_right_femur"):
        return load_bonevision_result(folder, orientation)


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        load_bonevision_result(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "upright_right_femur")
    elif globals().get("__file__") and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstruction.json")):
        # Run from inside an unzipped Slicer bundle: load the folder the script sits in.
        load_bonevision_result(os.path.dirname(os.path.abspath(__file__)))
