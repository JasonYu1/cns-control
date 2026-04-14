import numpy as np
from pymmcore_plus import CMMCorePlus
import time
from tqdm.auto import tqdm
from raman_control.andor import AndorSpectraCollector


def autofocus(plot=True, volts=[0, 0]):
    core = CMMCorePlus.instance()
    focusZ = core.getPosition()
    core.stopSequenceAcquisition()
    core.setExposure(0.1)
    # coarse_Z = np.linspace(-5, 5, 5)
    coarse_Z = np.linspace(-10, 10, 10)
    coarse_imgs = []
    core.setConfig("Channel", "RM")
    core.setShutterOpen("Fluoshutter", True)
    core.setAutoShutter(False)
    core.waitForSystem()
    collector = AndorSpectraCollector.instance()
    _ = collector.collect_spectra_volts([volts, volts], 0.1)
    for z in tqdm(coarse_Z):
        core.setPosition(focusZ + z)
        core.waitForSystem()
        time.sleep(0.25)
        # volts = transformer.BF_to_volts(points_layer.data/ [1024, 1344],max_volts=2)
        # spec = collect_spectra(volts, 1)
        coarse_imgs.append(core.snap())
        # coarse_spec.append(spec)
    coarse_imgs = np.asarray(coarse_imgs)
    # coarse_spec = np.asarray(coarse_spec)

    coarse_intensities = coarse_imgs.sum(axis=1).sum(axis=1)
    coarse_idx = np.argmax(coarse_intensities)

    fine_Z = coarse_Z[coarse_idx] + np.linspace(-5, 5, 20)
    fine_imgs = []
    for z in tqdm(fine_Z):
        core.setPosition(focusZ + z)
        core.waitForSystem()
        time.sleep(0.25)
        # volts = transformer.BF_to_volts(points_layer.data/ [1024, 1344],max_volts=2)
        # spec = collect_spectra(volts, 1)
        fine_imgs.append(core.snap())
        # spectra.append(spec)
    fine_imgs = np.asarray(fine_imgs)
    core.setAutoShutter(True)
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(coarse_Z, coarse_imgs.sum(axis=1).sum(axis=1), "o-")
        plt.plot(fine_Z, fine_imgs.sum(axis=1).sum(axis=1), "o-")
    fine_idx = np.argmax(fine_imgs.sum(axis=1).sum(axis=1))
    max_laser_offset = fine_Z[fine_idx]
    rm_offset = max_laser_offset - 6.8
    return focusZ + rm_offset, focusZ + max_laser_offset
